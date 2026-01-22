from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
from ansible_collections.community.vmware.plugins.module_utils.vmware import (PyVmomi, find_dvs_by_name, find_dvspg_by_name)
def get_managed_object(self, object_name=None, object_type=None):
    managed_object = None
    if not all([object_type, object_name]):
        return managed_object
    if object_type == 'VirtualMachine':
        managed_object = self.pyv.get_vm_or_template(object_name)
    if object_type == 'Folder':
        managed_object = self.pyv.find_folder_by_name(object_name)
    if object_type == 'Datacenter':
        managed_object = self.pyv.find_datacenter_by_name(object_name)
    if object_type == 'Datastore':
        managed_object = self.pyv.find_datastore_by_name(object_name)
    if object_type == 'DatastoreCluster':
        managed_object = self.pyv.find_datastore_cluster_by_name(object_name)
        self.object_type = 'StoragePod'
    if object_type == 'ClusterComputeResource':
        managed_object = self.pyv.find_cluster_by_name(object_name)
    if object_type == 'ResourcePool':
        managed_object = self.pyv.find_resource_pool_by_name(object_name)
    if object_type == 'HostSystem':
        managed_object = self.pyv.find_hostsystem_by_name(object_name)
    if object_type == 'DistributedVirtualSwitch':
        managed_object = find_dvs_by_name(self.pyv.content, object_name)
        self.object_type = 'VmwareDistributedVirtualSwitch'
    if object_type == 'DistributedVirtualPortgroup':
        dvs_name, pg_name = object_name.split(':', 1)
        dv_switch = find_dvs_by_name(self.pyv.content, dvs_name)
        if dv_switch is None:
            self.module.fail_json(msg='A distributed virtual switch with name %s does not exist' % dvs_name)
        managed_object = find_dvspg_by_name(dv_switch, pg_name)
    return managed_object