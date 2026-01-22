from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_service_policy(self, current, modify):
    api = 'network/ip/service-policies/%s' % current['uuid']
    modify_copy = dict(modify)
    body = {}
    for key in modify:
        if key in ('services',):
            body[key] = modify_copy.pop(key)
    if modify_copy:
        msg = 'Error: attributes not supported in modify: %s' % modify_copy
        self.module.fail_json(msg=msg)
    if not body:
        msg = 'Error: nothing to change - modify called with: %s' % modify
        self.module.fail_json(msg=msg)
    dummy, error = rest_generic.patch_async(self.rest_api, api, None, body)
    if error:
        msg = 'Error in modify_service_policy: %s' % error
        self.module.fail_json(msg=msg)