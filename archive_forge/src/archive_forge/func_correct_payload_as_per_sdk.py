from __future__ import absolute_import, division, print_function
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def correct_payload_as_per_sdk(self, payload, nfs_details=None):
    """ Correct payload keys as required by SDK

        :param payload: Payload used for create/modify operation
        :type payload: dict
        :param nfs_details: NFS details
        :type nfs_details: dict
        :return: Payload required by SDK
        :rtype: dict
        """
    ouput_host_param = self.host_param_mapping.values()
    if set(payload.keys()) & set(ouput_host_param):
        if not nfs_details or (nfs_details and nfs_details['export_option'] != 1):
            payload['export_option'] = 1
        if 'read_write_root_hosts_string' in payload:
            payload['root_access_hosts_string'] = payload.pop('read_write_root_hosts_string')
    return payload