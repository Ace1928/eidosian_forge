from __future__ import absolute_import, division, print_function
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver, zapis_svm
def create_body_contents(self, modify=None):
    keys_to_modify = self.parameters.keys() if modify is None else modify.keys()
    protocols_to_modify = self.parameters.get('services', {}) if modify is None else modify.get('services', {})
    simple_keys = ['name', 'language', 'ipspace', 'snapshot_policy', 'subtype', 'comment']
    if self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 9, 1):
        simple_keys.append('max_volumes')
    body = dict(((key, self.parameters[key]) for key in simple_keys if self.parameters.get(key) and key in keys_to_modify))
    if modify and 'admin_state' in keys_to_modify:
        body['state'] = self.parameters['admin_state']
    if 'aggr_list' in keys_to_modify:
        body['aggregates'] = [{'name': aggr} for aggr in self.parameters['aggr_list']]
    if 'certificate' in keys_to_modify:
        body['certificate'] = modify['certificate']
    allowed_protocols = {}
    for protocol, config in protocols_to_modify.items():
        if not config:
            continue
        acopy = self.na_helper.filter_out_none_entries(config)
        if modify is not None:
            acopy.pop('enabled', None)
        if not self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 9, 1):
            allowed = acopy.pop('allowed', None)
            if allowed is not None:
                allowed_protocols[protocol] = allowed
        if acopy:
            body[protocol] = acopy
    return (body, allowed_protocols)