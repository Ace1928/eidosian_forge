from __future__ import absolute_import, division, print_function
import re
from random import randint
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, \
from ansible_collections.community.vmware.plugins.module_utils.vm_device_helper import PyVmomiDeviceHelper
def ensure_disks(self, vm_obj=None):
    """
        Manage internal state of virtual machine disks
        Args:
            vm_obj: Managed object of virtual machine

        """
    self.vm = vm_obj
    vm_files_datastore = self.vm.config.files.vmPathName.split(' ')[0].strip('[]')
    disk_data = self.sanitize_disk_inputs()
    ctl_changed = False
    disk_change_list = list()
    results = dict(changed=False, disk_data=None, disk_changes=dict())
    new_added_disk_ctl = list()
    sharesval = {'low': 500, 'normal': 1000, 'high': 2000}
    for disk in disk_data:
        ctl_found = False
        for new_ctl in new_added_disk_ctl:
            if new_ctl['controller_type'] == disk['controller_type'] and new_ctl['controller_number'] == disk['controller_number']:
                ctl_found = True
                break
        if not ctl_found:
            for device in self.vm.config.hardware.device:
                if isinstance(device, self.device_helper.disk_ctl_device_type[disk['controller_type']]):
                    if device.busNumber == disk['controller_number']:
                        ctl_found = True
                        break
        if not ctl_found and disk['state'] == 'present':
            if disk['controller_type'] in self.device_helper.scsi_device_type.keys():
                ctl_spec = self.device_helper.create_scsi_controller(disk['controller_type'], disk['controller_number'], disk['bus_sharing'])
            elif disk['controller_type'] == 'sata':
                ctl_spec = self.device_helper.create_sata_controller(disk['controller_number'])
            elif disk['controller_type'] == 'nvme':
                ctl_spec = self.device_helper.create_nvme_controller(disk['controller_number'])
            new_added_disk_ctl.append({'controller_type': disk['controller_type'], 'controller_number': disk['controller_number']})
            ctl_changed = True
            self.config_spec.deviceChange.append(ctl_spec)
        elif not ctl_found and disk['state'] == 'absent':
            self.module.fail_json(msg="Not found 'controller_type': '%s', 'controller_number': '%s', so can not remove this disk, please make sure 'controller_type' and 'controller_number' are correct." % (disk['controller_type'], disk['controller_number']))
    if ctl_changed:
        self.reconfigure_vm(self.config_spec, 'Disk Controller')
        self.config_spec = vim.vm.ConfigSpec()
        self.config_spec.deviceChange = []
    for disk in disk_data:
        disk_found = False
        update_io = False
        disk_change = False
        ctl_found = False
        for device in self.vm.config.hardware.device:
            if isinstance(device, self.device_helper.disk_ctl_device_type[disk['controller_type']]) and device.busNumber == disk['controller_number']:
                for disk_key in device.device:
                    disk_device = self.find_disk_by_key(disk_key, disk['disk_unit_number'])
                    if disk_device is not None:
                        disk_found = True
                        if disk['state'] == 'present':
                            disk_spec = vim.vm.device.VirtualDeviceSpec()
                            disk_spec.device = disk_device
                            if 'iolimit' in disk:
                                if disk['iolimit']['limit'] != disk_spec.device.storageIOAllocation.limit:
                                    update_io = True
                                if 'shares' in disk['iolimit']:
                                    if disk['iolimit']['shares']['level'] != 'custom' and sharesval.get(disk['iolimit']['shares']['level'], 0) != disk_spec.device.storageIOAllocation.shares.shares or (disk['iolimit']['shares']['level'] == 'custom' and disk['iolimit']['shares']['level_value'] != disk_spec.device.storageIOAllocation.shares.shares):
                                        update_io = True
                                if update_io:
                                    disk_spec.operation = vim.vm.device.VirtualDeviceSpec.Operation.edit
                                    disk_spec = self.get_ioandshares_diskconfig(disk_spec, disk)
                                    disk_change = True
                            if disk['disk_type'] != 'rdm':
                                if disk['size'] < disk_spec.device.capacityInKB:
                                    self.module.fail_json(msg='Given disk size at disk index [%s] is smaller than found (%d < %d). Reducing disks is not allowed.' % (disk['disk_index'], disk['size'], disk_spec.device.capacityInKB))
                                if disk['size'] != disk_spec.device.capacityInKB:
                                    disk_spec.operation = vim.vm.device.VirtualDeviceSpec.Operation.edit
                                    if disk['disk_type'] != 'vpmemdisk':
                                        disk_spec = self.get_ioandshares_diskconfig(disk_spec, disk)
                                    disk_spec.device.capacityInKB = disk['size']
                                    disk_change = True
                            if disk_change:
                                self.config_spec.deviceChange.append(disk_spec)
                                disk_change_list.append(disk_change)
                                results['disk_changes'][disk['disk_index']] = 'Disk reconfigured.'
                        elif disk['state'] == 'absent':
                            disk_spec = vim.vm.device.VirtualDeviceSpec()
                            disk_spec.operation = vim.vm.device.VirtualDeviceSpec.Operation.remove
                            if disk['destroy'] is True:
                                disk_spec.fileOperation = vim.vm.device.VirtualDeviceSpec.FileOperation.destroy
                            disk_spec.device = disk_device
                            self.config_spec.deviceChange.append(disk_spec)
                            disk_change = True
                            disk_change_list.append(disk_change)
                            results['disk_changes'][disk['disk_index']] = 'Disk deleted.'
                        break
                if disk_found:
                    break
                if not disk_found and disk['state'] == 'present':
                    disk_spec = self.create_disk(device.key, disk)
                    if disk['disk_type'] == 'rdm':
                        if disk['filename'] is not None and disk['cluster_disk'] is True:
                            disk_spec.device.backing.fileName = disk['filename']
                        else:
                            disk_spec.fileOperation = vim.vm.device.VirtualDeviceSpec.FileOperation.create
                    else:
                        if disk['filename'] is None:
                            if disk['datastore_cluster'] is not None:
                                datastore_name = self.get_recommended_datastore(datastore_cluster_obj=disk['datastore_cluster'], disk_spec_obj=disk_spec)
                                disk['datastore'] = find_obj(self.content, [vim.Datastore], datastore_name)
                            disk_spec.fileOperation = vim.vm.device.VirtualDeviceSpec.FileOperation.create
                            disk_spec.device.capacityInKB = disk['size']
                            if disk['datastore'] is not None and disk['datastore'].name != vm_files_datastore:
                                disk_spec.device.backing.datastore = disk['datastore']
                                disk_spec.device.backing.fileName = '[%s] %s/%s_%s_%s_%s.vmdk' % (disk['datastore'].name, self.vm.name, self.vm.name, device.key, str(disk['disk_unit_number']), str(randint(1, 10000)))
                        elif disk['filename'] is not None:
                            disk_spec.device.backing.fileName = disk['filename']
                        disk_spec = self.get_ioandshares_diskconfig(disk_spec, disk)
                    self.config_spec.deviceChange.append(disk_spec)
                    disk_change = True
                    disk_change_list.append(disk_change)
                    results['disk_changes'][disk['disk_index']] = 'Disk created.'
                    break
        if disk_change:
            self.reconfigure_vm(self.config_spec, 'disks')
            self.config_spec = vim.vm.ConfigSpec()
            self.config_spec.deviceChange = []
    if any(disk_change_list):
        results['changed'] = True
    results['disk_data'] = self.device_helper.gather_disk_info(self.vm)
    self.module.exit_json(**results)