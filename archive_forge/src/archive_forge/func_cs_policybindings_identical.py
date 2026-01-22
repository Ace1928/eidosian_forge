from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.netscaler.netscaler import (
def cs_policybindings_identical(client, module):
    log('Checking policy bindings identical')
    actual_bindings = get_actual_policybindings(client, module)
    configured_bindings = get_configured_policybindings(client, module)
    actual_keyset = set(actual_bindings.keys())
    configured_keyset = set(configured_bindings.keys())
    if len(actual_keyset ^ configured_keyset) > 0:
        return False
    for key in actual_bindings.keys():
        configured_binding_proxy = configured_bindings[key]
        actual_binding_object = actual_bindings[key]
        if not configured_binding_proxy.has_equal_attributes(actual_binding_object):
            return False
    return True