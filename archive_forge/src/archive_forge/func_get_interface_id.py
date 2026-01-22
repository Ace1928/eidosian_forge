from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, json
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import MerakiModule, meraki_argument_spec
def get_interface_id(meraki, data, name):
    for interface in data:
        if interface['name'] == name:
            return interface['interfaceId']
    return None