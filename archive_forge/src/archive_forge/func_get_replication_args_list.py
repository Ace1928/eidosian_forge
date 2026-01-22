from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_replication_args_list(replication_params):
    """Returns the replication args for payload"""
    replication_args_list = {}
    if replication_params['replication_name']:
        replication_args_list['replication_name'] = replication_params['replication_name']
    if 'replication_mode' in replication_params and replication_params['replication_mode'] == 'asynchronous':
        replication_args_list['max_time_out_of_sync'] = replication_params['rpo']
    else:
        replication_args_list['max_time_out_of_sync'] = -1
    return replication_args_list