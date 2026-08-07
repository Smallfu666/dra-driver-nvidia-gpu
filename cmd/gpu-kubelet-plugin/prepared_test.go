/*
Copyright The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package main

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/utils/ptr"
)

func makeGpu(uuid, deviceName string) PreparedDevice {
	return PreparedDevice{Gpu: &PreparedGpu{
		Info:   &GpuInfo{UUID: uuid},
		Device: &CheckpointedDevice{DeviceName: deviceName, PoolName: "pool-gpu"},
	}}
}

func makeMig(migUUID, deviceName string) PreparedDevice {
	return PreparedDevice{Mig: &PreparedMigDevice{
		Concrete: &MigLiveTuple{MigUUID: migUUID},
		Device:   &CheckpointedDevice{DeviceName: deviceName, PoolName: "pool-mig"},
	}}
}

func makeVfio(uuid, deviceName string) PreparedDevice {
	return PreparedDevice{Vfio: &PreparedVfioDevice{
		Info:   &VfioDeviceInfo{UUID: uuid},
		Device: &CheckpointedDevice{DeviceName: deviceName, PoolName: "pool-vfio"},
	}}
}

func TestPreparedDeviceType(t *testing.T) {
	assert.Equal(t, GpuDeviceType, makeGpu("GPU-1", "g0").Type())
	assert.Equal(t, PreparedMigDeviceType, makeMig("MIG-1", "m0").Type())
	assert.Equal(t, VfioDeviceType, makeVfio("VF-1", "v0").Type())
	assert.Equal(t, UnknownDeviceType, PreparedDevice{}.Type())
}

func TestPreparedDeviceCanonicalName(t *testing.T) {
	g, m, v := makeGpu("GPU-1", "gpu-name"), makeMig("MIG-1", "mig-name"), makeVfio("VF-1", "vfio-name")
	assert.Equal(t, "gpu-name", (&g).CanonicalName())
	assert.Equal(t, "mig-name", (&m).CanonicalName())
	assert.Equal(t, "vfio-name", (&v).CanonicalName())

	require.Panics(t, func() { _ = (&PreparedDevice{}).CanonicalName() })
}

func TestPreparedDeviceListFilters(t *testing.T) {
	// Interleaved types prove filtering picks the right subset in input order.
	list := PreparedDeviceList{
		makeGpu("GPU-1", "g0"),
		makeMig("MIG-1", "m0"),
		makeVfio("VF-1", "v0"),
		makeGpu("GPU-2", "g1"),
		makeMig("MIG-2", "m1"),
	}

	gpus := list.Gpus()
	require.Len(t, gpus, 2)
	assert.Equal(t, "GPU-1", gpus[0].Gpu.Info.UUID)
	assert.Equal(t, "GPU-2", gpus[1].Gpu.Info.UUID)

	migs := list.MigDevices()
	require.Len(t, migs, 2)
	assert.Equal(t, "MIG-1", migs[0].Mig.Concrete.MigUUID)

	require.Len(t, list.VfioDevices(), 1)

	var empty PreparedDeviceList
	assert.Empty(t, empty.Gpus())
	assert.Empty(t, empty.MigDevices())
	assert.Empty(t, empty.VfioDevices())
}

func TestGetDevicesAndNames(t *testing.T) {
	// GetDevices is a no-op cast to kubeletplugin.Device.
	g := &PreparedDeviceGroup{Devices: PreparedDeviceList{
		makeGpu("GPU-1", "g0"), makeMig("MIG-1", "m0"), makeVfio("VF-1", "v0"),
	}}
	devs := g.GetDevices()
	require.Len(t, devs, 3)
	assert.Equal(t, "g0", devs[0].DeviceName)
	assert.Equal(t, "pool-gpu", devs[0].PoolName)
	assert.Equal(t, []DeviceName{"g0", "m0", "v0"}, g.GetDeviceNames())

	d := PreparedDevices{
		g,
		&PreparedDeviceGroup{Devices: PreparedDeviceList{makeGpu("GPU-2", "g1")}},
	}
	assert.Equal(t, []DeviceName{"g0", "m0", "v0", "g1"}, d.GetDeviceNames())
}

// Per-type concatenation of these is NOT globally sorted, so UUIDs()'s final
// sort is load-bearing.
var wantSorted = []string{"a-gpu", "b-vfio", "m-mig", "u-gpu", "z-vfio"}

func TestPreparedDeviceListUUIDs(t *testing.T) {
	list := PreparedDeviceList{
		makeGpu("u-gpu", "g0"),
		makeVfio("b-vfio", "v0"),
		makeMig("m-mig", "m0"),
		makeGpu("a-gpu", "g1"),
		makeVfio("z-vfio", "v1"),
	}
	assert.Equal(t, []string{"a-gpu", "u-gpu"}, list.GpuUUIDs())
	assert.Equal(t, []string{"m-mig"}, list.MigDeviceUUIDs())
	assert.Equal(t, []string{"b-vfio", "z-vfio"}, list.VfioDeviceUUIDs())
	assert.Equal(t, wantSorted, list.UUIDs())
}

func TestPreparedDevicesUUIDsAcrossGroups(t *testing.T) {
	// Per-type sets span both groups out of order, so the cross-group
	// aggregation sorts are load-bearing (distinct from the list-level path).
	d := PreparedDevices{
		&PreparedDeviceGroup{Devices: PreparedDeviceList{
			makeGpu("u-gpu", "g0"), makeVfio("z-vfio", "v0"),
		}},
		&PreparedDeviceGroup{Devices: PreparedDeviceList{
			makeMig("m-mig", "m0"), makeGpu("a-gpu", "g1"), makeVfio("b-vfio", "v1"),
		}},
	}
	assert.Equal(t, []string{"a-gpu", "u-gpu"}, d.GpuUUIDs())
	assert.Equal(t, []string{"b-vfio", "z-vfio"}, d.VfioDeviceUUIDs())
	assert.Equal(t, wantSorted, d.UUIDs())

	var empty PreparedDevices
	assert.Empty(t, empty.UUIDs())
}

func TestGetNonAdminDevices(t *testing.T) {
	// Keeps devices on our driver that are not admin-access.
	claim := &PreparedClaim{Status: resourceapi.ResourceClaimStatus{
		Allocation: &resourceapi.AllocationResult{Devices: resourceapi.DeviceAllocationResult{
			Results: []resourceapi.DeviceRequestAllocationResult{
				{Driver: DriverName, Device: "dev-nil-admin", AdminAccess: nil},
				{Driver: DriverName, Device: "dev-false-admin", AdminAccess: ptr.To(false)},
				{Driver: DriverName, Device: "dev-true-admin", AdminAccess: ptr.To(true)},
				{Driver: "other.driver.com", Device: "dev-other-driver", AdminAccess: nil},
			},
		}},
	}}
	assert.Equal(t, map[string]struct{}{
		"dev-nil-admin":   {},
		"dev-false-admin": {},
	}, claim.GetNonAdminDevices())

	emptyClaim := &PreparedClaim{Status: resourceapi.ResourceClaimStatus{
		Allocation: &resourceapi.AllocationResult{},
	}}
	assert.Empty(t, emptyClaim.GetNonAdminDevices())
}
