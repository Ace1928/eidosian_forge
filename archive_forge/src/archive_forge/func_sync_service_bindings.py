from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.netscaler.netscaler import (
import copy
def sync_service_bindings(client, module):
    log('sync_service_bindings')
    actual_bindings = get_actual_service_bindings(client, module)
    configured_bindigns = get_configured_service_bindings(client, module)
    delete_keys = list(set(actual_bindings.keys()) - set(configured_bindigns.keys()))
    for key in delete_keys:
        log('Deleting service binding %s' % key)
        actual_bindings[key].servicegroupname = ''
        actual_bindings[key].delete(client, actual_bindings[key])
    add_keys = list(set(configured_bindigns.keys()) - set(actual_bindings.keys()))
    for key in add_keys:
        log('Adding service binding %s' % key)
        configured_bindigns[key].add()
    modify_keys = list(set(configured_bindigns.keys()) & set(actual_bindings.keys()))
    for key in modify_keys:
        if not configured_bindigns[key].has_equal_attributes(actual_bindings[key]):
            log('Updating service binding %s' % key)
            actual_bindings[key].servicegroupname = ''
            actual_bindings[key].delete(client, actual_bindings[key])
            configured_bindigns[key].add()