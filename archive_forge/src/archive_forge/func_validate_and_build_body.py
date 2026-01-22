from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def validate_and_build_body(self, modify=None):
    if modify is None:
        required_keys = set(['provider_type', 'server', 'container', 'access_key'])
        if not required_keys.issubset(set(self.parameters.keys())):
            self.module.fail_json(msg='Error provisioning object store %s: one of the following parameters are missing %s' % (self.parameters['name'], ', '.join(required_keys)))
    if not self.use_rest:
        return None
    params = self.parameters if modify is None else modify
    body = {}
    for key in self.rest_all_fields:
        if params.get(key) is not None:
            body[key] = params[key]
    if not modify and 'owner' not in body:
        body['owner'] = 'fabricpool'
    if modify and 'owner' in body:
        self.module.fail_json(msg='Error modifying object store, owner cannot be changed.  Found: %s.' % body['owner'])
    return body