from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.infinidat.infinibox.plugins.module_utils.infinibox import (
@api_wrapper
def create_empty_network_space(module, system):
    """ Create an empty network space """
    network_space_name = module.params['name']
    service = module.params['service']
    rate_limit = module.params['rate_limit']
    mtu = module.params['mtu']
    network_config = {'netmask': module.params['netmask'], 'network': module.params['network'], 'default_gateway': module.params['default_gateway']}
    interfaces = module.params['interfaces']
    net_create_url = 'network/spaces'
    net_create_data = {'name': network_space_name, 'service': service, 'network_config': network_config, 'interfaces': interfaces}
    if rate_limit:
        net_create_data['rate_limit'] = rate_limit
    if mtu:
        net_create_data['mtu'] = mtu
    try:
        system.api.post(path=net_create_url, data=net_create_data)
    except APICommandFailed as err:
        module.fail_json(msg=f'Cannot create empty network space {network_space_name}: {err}')