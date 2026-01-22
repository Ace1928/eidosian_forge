from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.netscaler.netscaler import (
def get_actual_domain_bindings(client, module):
    log('get_actual_domain_bindings')
    actual_domain_bindings = {}
    if gslbvserver_domain_binding.count(client, name=module.params['name']) != 0:
        fetched_domain_bindings = gslbvserver_domain_binding.get(client, name=module.params['name'])
        for binding in fetched_domain_bindings:
            complete_missing_attributes(binding, gslbvserver_domain_binding_rw_attrs, fill_value=None)
            actual_domain_bindings[binding.domainname] = binding
    return actual_domain_bindings