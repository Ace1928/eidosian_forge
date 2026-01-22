from __future__ import absolute_import, division, print_function
import time
import traceback
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def quota_entry_modify_rest(self, modify_quota):
    """
        quota_entry_modify with rest API.
        User mapping cannot be turned on for multiuser quota rules.
        """
    if not self.use_rest:
        return self.quota_entry_modify(modify_quota)
    body = {}
    if 'disk_limit' in modify_quota:
        body['space.hard_limit'] = modify_quota['disk_limit']
    if 'file_limit' in modify_quota:
        body['files.hard_limit'] = modify_quota['file_limit']
    if 'soft_disk_limit' in modify_quota:
        body['space.soft_limit'] = modify_quota['soft_disk_limit']
    if 'soft_file_limit' in modify_quota:
        body['files.soft_limit'] = modify_quota['soft_file_limit']
    if 'perform_user_mapping' in modify_quota:
        body['user_mapping'] = modify_quota['perform_user_mapping']
    api = 'storage/quota/rules'
    dummy, error = rest_generic.patch_async(self.rest_api, api, self.quota_uuid, body)
    if error is not None:
        if '5308567' in error:
            self.form_warn_msg_rest('modify', '5308567')
        else:
            self.module.fail_json(msg='Error on modifying quotas rule: %s' % error)