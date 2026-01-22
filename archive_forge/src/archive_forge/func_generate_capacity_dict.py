from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
from datetime import datetime
def generate_capacity_dict(module, blade):
    capacity_info = {}
    api_version = blade.api_version.list_versions().versions
    if SPACE_API_VERSION in api_version:
        blade2 = get_system(module)
        total_cap = list(blade2.get_arrays_space().items)[0]
        file_cap = list(blade2.get_arrays_space(type='file-system').items)[0]
        object_cap = list(blade2.get_arrays_space(type='object-store').items)[0]
        capacity_info['total'] = total_cap.space.capacity
        capacity_info['aggregate'] = {'data_reduction': total_cap.space.data_reduction, 'snapshots': total_cap.space.snapshots, 'total_physical': total_cap.space.total_physical, 'unique': total_cap.space.unique, 'virtual': total_cap.space.virtual, 'total_provisioned': total_cap.space.total_provisioned, 'available_provisioned': total_cap.space.available_provisioned, 'available_ratio': total_cap.space.available_ratio, 'destroyed': total_cap.space.destroyed, 'destroyed_virtual': total_cap.space.destroyed_virtual}
        capacity_info['file-system'] = {'data_reduction': file_cap.space.data_reduction, 'snapshots': file_cap.space.snapshots, 'total_physical': file_cap.space.total_physical, 'unique': file_cap.space.unique, 'virtual': file_cap.space.virtual, 'total_provisioned': total_cap.space.total_provisioned, 'available_provisioned': total_cap.space.available_provisioned, 'available_ratio': total_cap.space.available_ratio, 'destroyed': total_cap.space.destroyed, 'destroyed_virtual': total_cap.space.destroyed_virtual}
        capacity_info['object-store'] = {'data_reduction': object_cap.space.data_reduction, 'snapshots': object_cap.space.snapshots, 'total_physical': object_cap.space.total_physical, 'unique': object_cap.space.unique, 'virtual': file_cap.space.virtual, 'total_provisioned': total_cap.space.total_provisioned, 'available_provisioned': total_cap.space.available_provisioned, 'available_ratio': total_cap.space.available_ratio, 'destroyed': total_cap.space.destroyed, 'destroyed_virtual': total_cap.space.destroyed_virtual}
    else:
        total_cap = blade.arrays.list_arrays_space()
        file_cap = blade.arrays.list_arrays_space(type='file-system')
        object_cap = blade.arrays.list_arrays_space(type='object-store')
        capacity_info['total'] = total_cap.items[0].capacity
        capacity_info['aggregate'] = {'data_reduction': total_cap.items[0].space.data_reduction, 'snapshots': total_cap.items[0].space.snapshots, 'total_physical': total_cap.items[0].space.total_physical, 'unique': total_cap.items[0].space.unique, 'virtual': total_cap.items[0].space.virtual}
        capacity_info['file-system'] = {'data_reduction': file_cap.items[0].space.data_reduction, 'snapshots': file_cap.items[0].space.snapshots, 'total_physical': file_cap.items[0].space.total_physical, 'unique': file_cap.items[0].space.unique, 'virtual': file_cap.items[0].space.virtual}
        capacity_info['object-store'] = {'data_reduction': object_cap.items[0].space.data_reduction, 'snapshots': object_cap.items[0].space.snapshots, 'total_physical': object_cap.items[0].space.total_physical, 'unique': object_cap.items[0].space.unique, 'virtual': file_cap.items[0].space.virtual}
    return capacity_info