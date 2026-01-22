from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.netscaler.netscaler import (
import copy
def servicegroup_bindings_identical(client, module):
    log('servicegroup_bindings_identical')
    configured_servicegroup_bindings = get_configured_servicegroup_bindings(client, module)
    servicegroup_bindings = get_actual_servicegroup_bindings(client, module)
    configured_keyset = set(configured_servicegroup_bindings.keys())
    service_keyset = set(servicegroup_bindings.keys())
    log('len %s' % len(configured_keyset ^ service_keyset))
    if len(configured_keyset ^ service_keyset) > 0:
        return False
    for key in configured_servicegroup_bindings.keys():
        conf = configured_servicegroup_bindings[key]
        serv = servicegroup_bindings[key]
        log('sg diff %s' % conf.diff_object(serv))
        if not conf.has_equal_attributes(serv):
            return False
    return True