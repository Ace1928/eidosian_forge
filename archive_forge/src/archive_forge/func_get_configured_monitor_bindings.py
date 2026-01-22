from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import copy
from ansible_collections.community.network.plugins.module_utils.network.netscaler.netscaler import (ConfigProxy, get_nitro_client, netscaler_common_arguments,
def get_configured_monitor_bindings(client, module):
    log('Entering get_configured_monitor_bindings')
    bindings = {}
    if 'monitorbindings' in module.params and module.params['monitorbindings'] is not None:
        for binding in module.params['monitorbindings']:
            readwrite_attrs = ['monitorname', 'servicegroupname', 'weight']
            readonly_attrs = []
            attribute_values_dict = copy.deepcopy(binding)
            attribute_values_dict['servicegroupname'] = module.params['servicegroupname']
            binding_proxy = ConfigProxy(actual=lbmonitor_servicegroup_binding(), client=client, attribute_values_dict=attribute_values_dict, readwrite_attrs=readwrite_attrs, readonly_attrs=readonly_attrs)
            key = attribute_values_dict['monitorname']
            bindings[key] = binding_proxy
    return bindings