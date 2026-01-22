from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import copy
from ansible_collections.community.network.plugins.module_utils.network.netscaler.netscaler import (ConfigProxy, get_nitro_client, netscaler_common_arguments,
def get_configured_service_members(client, module):
    log('get_configured_service_members')
    readwrite_attrs = ['servicegroupname', 'ip', 'port', 'state', 'hashid', 'serverid', 'servername', 'customserverid', 'weight']
    readonly_attrs = ['delay', 'statechangetimesec', 'svrstate', 'tickssincelaststatechange', 'graceful']
    members = []
    if module.params['servicemembers'] is None:
        return members
    for config in module.params['servicemembers']:
        config = copy.deepcopy(config)
        config['servicegroupname'] = module.params['servicegroupname']
        member_proxy = ConfigProxy(actual=servicegroup_servicegroupmember_binding(), client=client, attribute_values_dict=config, readwrite_attrs=readwrite_attrs, readonly_attrs=readonly_attrs)
        members.append(member_proxy)
    return members