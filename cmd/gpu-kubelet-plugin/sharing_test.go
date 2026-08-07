/*
 * Copyright 2026 The Kubernetes Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package main

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
)

// mockFileChecker implements fileChecker for tests.
// existingPath is the single path Stat should report as existing; empty means nothing exists.
type mockFileChecker struct {
	existingPath string
}

func (m *mockFileChecker) Stat(path string) error {
	if path == m.existingPath {
		return nil
	}
	return errors.New("not found")
}

func TestSetMpsShmMountPath(t *testing.T) {
	testCases := map[string]struct {
		existingPath      string
		expectedMountPath string
	}{
		// /dev/shm exists under the driver root → daemon uses chroot → shm at <driverRootMountDir>/dev/shm.
		"dev/shm exists under driver root": {
			existingPath:      filepath.Join(driverRootMountDir, "dev", "shm"),
			expectedMountPath: filepath.Join(driverRootMountDir, "dev", "shm"),
		},
		// /dev/shm not present under driver root (e.g. GKE COS) → daemon runs directly
		// in the container namespace → shm at /dev/shm.
		"dev/shm does not exist under driver root — case for GKE COS": {
			existingPath:      "",
			expectedMountPath: MpsDefaultShmMountPath,
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			checker := &mockFileChecker{existingPath: tc.existingPath}
			require.Equal(t, tc.expectedMountPath, setMpsShmMountPath(checker))
		})
	}
}

func TestRenderMpsControlDaemonDeploymentImagePullSettings(t *testing.T) {
	deployment, err := renderMpsControlDaemonDeployment(
		filepath.Join("..", "..", "templates", "mps-control-daemon.tmpl.yaml"),
		MpsControlDaemonTemplateData{
			NodeName:                  "node-a",
			MpsControlDaemonNamespace: "dra-driver-nvidia-gpu",
			MpsControlDaemonName:      "mps-control-daemon-test",
			CUDA_VISIBLE_DEVICES:      "GPU-0",
			NvidiaDriverRoot:          "/",
			MpsShmDirectory:           "/var/lib/kubelet/plugins/gpu.nvidia.com/mps/test/shm",
			MpsPipeDirectory:          "/var/lib/kubelet/plugins/gpu.nvidia.com/mps/test/pipe",
			MpsLogDirectory:           "/var/lib/kubelet/plugins/gpu.nvidia.com/mps/test/log",
			MpsImageName:              "registry.example.com/dra-driver:dev",
			MpsImagePullPolicy:        "Always",
			MpsImagePullSecretNames:   []string{"regcred", "mirrorcred"},
			MpsShmMountPath:           MpsDefaultShmMountPath,
		},
	)
	require.NoError(t, err)

	require.Equal(t, []corev1.LocalObjectReference{
		{Name: "regcred"},
		{Name: "mirrorcred"},
	}, deployment.Spec.Template.Spec.ImagePullSecrets)
	require.Len(t, deployment.Spec.Template.Spec.Containers, 1)
	require.Equal(t, corev1.PullAlways, deployment.Spec.Template.Spec.Containers[0].ImagePullPolicy)
}

// fakeUUIDProvider is a test UUIDProvider that returns pre-set UUID slices.
// Note: the real callers are expected to hand GetMpsControlDaemonID a provider
// whose UUIDs() returns a stable (sorted) order, because the daemon id is a hash
// of strings.Join(UUIDs(), ","); a different order yields a different id.
type fakeUUIDProvider struct {
	uuids    []string
	gpuUUIDs []string
	migUUIDs []string
}

func (f fakeUUIDProvider) UUIDs() []string          { return f.uuids }
func (f fakeUUIDProvider) GpuUUIDs() []string       { return f.gpuUUIDs }
func (f fakeUUIDProvider) MigDeviceUUIDs() []string { return f.migUUIDs }

// expectedDaemonID recomputes the id contract independently of production code:
// "<claimUID>-<first 5 hex chars of sha256(join(uuids, ","))>".
func expectedDaemonID(claimUID string, uuids []string) string {
	sum := sha256.Sum256([]byte(strings.Join(uuids, ",")))
	return claimUID + "-" + hex.EncodeToString(sum[:])[:5]
}

func TestGetMpsControlDaemonID(t *testing.T) {
	m := &MpsManager{}

	t.Run("pinned id matches externally-computed sha256", func(t *testing.T) {
		// sha256("GPU-A,GPU-B") begins with 0ae63 (verified with sha256sum).
		id := m.GetMpsControlDaemonID("claim-1", fakeUUIDProvider{uuids: []string{"GPU-A", "GPU-B"}})
		assert.Equal(t, "claim-1-0ae63", id)
		assert.Equal(t, expectedDaemonID("claim-1", []string{"GPU-A", "GPU-B"}), id)
	})

	t.Run("single uuid pinned", func(t *testing.T) {
		// sha256("GPU-A") begins with bb3ae.
		id := m.GetMpsControlDaemonID("claim-1", fakeUUIDProvider{uuids: []string{"GPU-A"}})
		assert.Equal(t, "claim-1-bb3ae", id)
	})

	t.Run("empty uuid list is handled", func(t *testing.T) {
		// sha256("") begins with e3b0c (the well-known empty-input digest).
		id := m.GetMpsControlDaemonID("claim-1", fakeUUIDProvider{uuids: nil})
		assert.Equal(t, "claim-1-e3b0c", id)
	})

	t.Run("deterministic: same input yields same id", func(t *testing.T) {
		devices := fakeUUIDProvider{uuids: []string{"GPU-A", "GPU-B"}}
		assert.Equal(t, m.GetMpsControlDaemonID("claim-1", devices), m.GetMpsControlDaemonID("claim-1", devices))
	})

	t.Run("different uuid sets yield different ids", func(t *testing.T) {
		a := m.GetMpsControlDaemonID("claim-1", fakeUUIDProvider{uuids: []string{"GPU-A", "GPU-B"}})
		b := m.GetMpsControlDaemonID("claim-1", fakeUUIDProvider{uuids: []string{"GPU-C"}})
		assert.NotEqual(t, a, b)
	})

	t.Run("different claim uid changes only the prefix", func(t *testing.T) {
		devices := fakeUUIDProvider{uuids: []string{"GPU-A", "GPU-B"}}
		assert.Equal(t, "claim-2-0ae63", m.GetMpsControlDaemonID("claim-2", devices))
	})
}

// newTestConfig builds a minimal *Config sufficient for the path-formatting
// logic in NewMpsManager / NewMpsControlDaemon. It needs no clientsets, NVML, or
// flags parsing — only the handful of flag fields those constructors read.
func newTestConfig() *Config {
	return &Config{
		flags: &Flags{
			nodeName:                    "node-a",
			namespace:                   "dra-ns",
			kubeletPluginsDirectoryPath: "/var/lib/kubelet/plugins",
		},
	}
}

func TestNewMpsManagerControlFilesRoot(t *testing.T) {
	// controlFilesRoot = <kubeletPluginsDirectoryPath>/<DriverName>/mps
	mgr := NewMpsManager(newTestConfig(), nil, "/run/nvidia/driver", MpsControlDaemonTemplatePath)
	require.NotNil(t, mgr)
	assert.Equal(t, filepath.Join("/var/lib/kubelet/plugins", DriverName, MpsControlFilesDirName), mgr.controlFilesRoot)
	assert.Equal(t, "/run/nvidia/driver", mgr.hostDriverRoot)
	assert.Equal(t, MpsControlDaemonTemplatePath, mgr.templatePath)
}

func TestNewMpsControlDaemonPaths(t *testing.T) {
	cfg := newTestConfig()
	mgr := NewMpsManager(cfg, nil, "/run/nvidia/driver", MpsControlDaemonTemplatePath)
	devices := fakeUUIDProvider{uuids: []string{"GPU-A", "GPU-B"}}

	d := mgr.NewMpsControlDaemon("claim-1", devices)
	require.NotNil(t, d)

	id := "claim-1-0ae63"
	root := mgr.controlFilesRoot + "/" + id
	assert.Equal(t, id, d.GetID())
	assert.Equal(t, id, d.id)
	assert.Equal(t, "node-a", d.nodeName)
	assert.Equal(t, "dra-ns", d.namespace)
	assert.Equal(t, "mps-control-daemon-"+id, d.name)
	assert.Equal(t, root, d.rootDir)
	assert.Equal(t, root+"/pipe", d.pipeDir)
	assert.Equal(t, root+"/shm", d.shmDir)
	assert.Equal(t, root+"/log", d.logDir)
	// The daemon retains the exact provider it was constructed with.
	assert.Equal(t, devices, d.devices)
	assert.Same(t, mgr, d.manager)
}

func TestGetCDIContainerEdits(t *testing.T) {
	d := &MpsControlDaemon{
		shmDir:  "/host/mps/claim-1/shm",
		pipeDir: "/host/mps/claim-1/pipe",
	}

	edits := d.GetCDIContainerEdits()
	require.NotNil(t, edits)
	require.NotNil(t, edits.ContainerEdits)

	// The pipe directory env var is always the fixed in-container path.
	assert.Equal(t, []string{"CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps"}, edits.Env)

	require.Len(t, edits.Mounts, 2)
	expectedOptions := []string{"rw", "nosuid", "nodev", "bind"}

	// shm mount: /dev/shm in the container <- shmDir on the host.
	assert.Equal(t, "/dev/shm", edits.Mounts[0].ContainerPath)
	assert.Equal(t, "/host/mps/claim-1/shm", edits.Mounts[0].HostPath)
	assert.Equal(t, expectedOptions, edits.Mounts[0].Options)

	// pipe mount: /tmp/nvidia-mps in the container <- pipeDir on the host.
	assert.Equal(t, "/tmp/nvidia-mps", edits.Mounts[1].ContainerPath)
	assert.Equal(t, "/host/mps/claim-1/pipe", edits.Mounts[1].HostPath)
	assert.Equal(t, expectedOptions, edits.Mounts[1].Options)
}

// minimalDeploymentTemplate renders a valid Deployment using a few template
// fields, so we can assert renderMpsControlDaemonDeployment wires them through.
const minimalDeploymentTemplate = `apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ .MpsControlDaemonName }}
  namespace: {{ .MpsControlDaemonNamespace }}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: {{ .MpsControlDaemonName }}
  template:
    metadata:
      labels:
        app: {{ .MpsControlDaemonName }}
    spec:
      containers:
      - name: mps-control-daemon
        image: test-image
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "{{ .CUDA_VISIBLE_DEVICES }}"
`

func writeTempTemplate(t *testing.T, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "tmpl.yaml")
	require.NoError(t, os.WriteFile(path, []byte(content), 0o600))
	return path
}

func TestRenderMpsControlDaemonDeploymentFromTemplate(t *testing.T) {
	path := writeTempTemplate(t, minimalDeploymentTemplate)

	deployment, err := renderMpsControlDaemonDeployment(path, MpsControlDaemonTemplateData{
		MpsControlDaemonName:      "mps-control-daemon-claim-1",
		MpsControlDaemonNamespace: "dra-ns",
		CUDA_VISIBLE_DEVICES:      "GPU-A,GPU-B",
	})
	require.NoError(t, err)
	require.NotNil(t, deployment)

	assert.Equal(t, "mps-control-daemon-claim-1", deployment.Name)
	assert.Equal(t, "dra-ns", deployment.Namespace)
	assert.Equal(t, map[string]string{"app": "mps-control-daemon-claim-1"}, deployment.Spec.Selector.MatchLabels)
	require.Len(t, deployment.Spec.Template.Spec.Containers, 1)
	c := deployment.Spec.Template.Spec.Containers[0]
	require.Len(t, c.Env, 1)
	assert.Equal(t, "CUDA_VISIBLE_DEVICES", c.Env[0].Name)
	assert.Equal(t, "GPU-A,GPU-B", c.Env[0].Value)
}

func TestRenderMpsControlDaemonDeploymentErrors(t *testing.T) {
	t.Run("non-existent template path", func(t *testing.T) {
		_, err := renderMpsControlDaemonDeployment(
			filepath.Join(t.TempDir(), "does-not-exist.yaml"),
			MpsControlDaemonTemplateData{},
		)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to parse template file")
	})

	t.Run("template renders non-object YAML", func(t *testing.T) {
		// A bare scalar is valid YAML but is not a Kubernetes object, so
		// unmarshalling into an unstructured object fails.
		path := writeTempTemplate(t, "just-a-scalar-{{ .MpsControlDaemonName }}\n")
		_, err := renderMpsControlDaemonDeployment(path, MpsControlDaemonTemplateData{MpsControlDaemonName: "x"})
		require.Error(t, err)
	})

	t.Run("template renders malformed YAML", func(t *testing.T) {
		// Unbalanced flow mapping -> YAML parse error.
		path := writeTempTemplate(t, "metadata: {name: {{ .MpsControlDaemonName }}\n")
		_, err := renderMpsControlDaemonDeployment(path, MpsControlDaemonTemplateData{MpsControlDaemonName: "x"})
		require.Error(t, err)
	})
}
