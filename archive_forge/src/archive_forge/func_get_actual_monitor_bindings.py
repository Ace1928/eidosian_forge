from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import copy
from ansible_collections.community.network.plugins.module_utils.network.netscaler.netscaler import (ConfigProxy, get_nitro_client, netscaler_common_arguments,
def get_actual_monitor_bindings(client, module):
    log('Entering get_actual_monitor_bindings')
    bindings = {}
    try:
        count = servicegroup_lbmonitor_binding.count(client, module.params['servicegroupname'])
    except nitro_exception as e:
        if e.errorcode == 258:
            return bindings
        else:
            raise
    if count == 0:
        return bindings
    for binding in servicegroup_lbmonitor_binding.get(client, module.params['servicegroupname']):
        log('Gettign actual monitor with name %s' % binding.monitor_name)
        key = binding.monitor_name
        bindings[key] = binding
    return bindings