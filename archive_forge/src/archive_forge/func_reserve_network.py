from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, json
from ansible.module_utils.urls import open_url
def reserve_network(self, network_id='', reserved_network_name='', reserved_network_description='', reserved_network_size='', reserved_network_family='4', reserved_network_type='lan', reserved_network_address=''):
    """
        Reserves the first available network of specified size from a given supernet
         <dt>network_name (required)</dt><dd>Name of the network</dd>
            <dt>description (optional)</dt><dd>Free description</dd>
            <dt>network_family (required)</dt><dd>Address family of the network. One of '4', '6', 'IPv4', 'IPv6', 'dual'</dd>
            <dt>network_address (optional)</dt><dd>Address of the new network. If not given, the first network available will be created.</dd>
            <dt>network_size (required)</dt><dd>Size of the new network in /&lt;prefix&gt; notation.</dd>
            <dt>network_type (required)</dt><dd>Type of network. One of 'supernet', 'lan', 'shared_lan'</dd>

        """
    method = 'post'
    resource_url = ''
    network_info = None
    if network_id is None or reserved_network_name is None or reserved_network_size is None:
        self.module.exit_json(msg="You must specify those options: 'network_id', 'reserved_network_name' and 'reserved_network_size'")
    if network_id:
        resource_url = 'networks/' + str(network_id) + '/reserve_network'
    if not reserved_network_family:
        reserved_network_family = '4'
    if not reserved_network_type:
        reserved_network_type = 'lan'
    payload_data = {'network_name': reserved_network_name, 'description': reserved_network_description, 'network_size': reserved_network_size, 'network_family': reserved_network_family, 'network_type': reserved_network_type, 'network_location': int(network_id)}
    if reserved_network_address:
        payload_data.update({'network_address': reserved_network_address})
    network_info = self._get_api_call_ansible_handler(method, resource_url, stat_codes=[200, 201], payload_data=payload_data)
    return network_info