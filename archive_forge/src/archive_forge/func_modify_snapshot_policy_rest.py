from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_snapshot_policy_rest(self, modify, current=None):
    """
        Modify snapshot policy with rest API.
        """
    if not self.use_rest:
        return self.modify_snapshot_policy(current)
    api = 'storage/snapshot-policies'
    body = {}
    if 'enabled' in modify:
        body['enabled'] = modify['enabled']
    if 'comment' in modify:
        body['comment'] = modify['comment']
    if body:
        dummy, error = rest_generic.patch_async(self.rest_api, api, current['uuid'], body)
        if error is not None:
            self.module.fail_json(msg='Error on modifying snapshot policy: %s' % error)