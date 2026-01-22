from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, json
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import (
def get_network_settings(meraki, net_id):
    path = meraki.construct_path('get_settings', net_id=net_id)
    response = meraki.request(path, method='GET')
    return response