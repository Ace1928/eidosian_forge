from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
from datetime import datetime
def generate_array_conn_dict(module, blade):
    array_conn_info = {}
    arraysv2 = {}
    api_version = blade.api_version.list_versions().versions
    arrays = blade.array_connections.list_array_connections()
    if NFS_POLICY_API_VERSION in api_version:
        bladev2 = get_system(module)
        arraysv2 = list(bladev2.get_array_connections().items)
    for arraycnt in range(0, len(arrays.items)):
        array = arrays.items[arraycnt].remote.name
        array_conn_info[array] = {'encrypted': arrays.items[arraycnt].encrypted, 'replication_addresses': arrays.items[arraycnt].replication_addresses, 'management_address': arrays.items[arraycnt].management_address, 'status': arrays.items[arraycnt].status, 'version': arrays.items[arraycnt].version, 'throttle': []}
        if arrays.items[arraycnt].encrypted:
            array_conn_info[array]['ca_certificate_group'] = arrays.items[arraycnt].ca_certificate_group.name
        for v2array in range(0, len(arraysv2)):
            if arraysv2[v2array].remote.name == array:
                array_conn_info[array]['throttle'] = {'default_limit': _bytes_to_human(arraysv2[v2array].throttle.default_limit), 'window_limit': _bytes_to_human(arraysv2[v2array].throttle.window_limit), 'window_start': _millisecs_to_time(arraysv2[v2array].throttle.window.start), 'window_end': _millisecs_to_time(arraysv2[v2array].throttle.window.end)}
    return array_conn_info