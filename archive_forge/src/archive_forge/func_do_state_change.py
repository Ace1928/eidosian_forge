from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import copy
from ansible_collections.community.network.plugins.module_utils.network.netscaler.netscaler import (ConfigProxy, get_nitro_client, netscaler_common_arguments,
def do_state_change(client, module, servicegroup_proxy):
    if module.params['disabled']:
        log('Disabling service')
        result = servicegroup.disable(client, servicegroup_proxy.actual)
    else:
        log('Enabling service')
        result = servicegroup.enable(client, servicegroup_proxy.actual)
    return result