from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule, json
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import MerakiModule, meraki_argument_spec
def network_factory(meraki, networks, nets):
    networks_new = []
    for n in networks:
        if 'network' in n and n['network'] is not None:
            networks_new.append({'id': meraki.get_net_id(org_name=meraki.params['org_name'], net_name=n['network'], data=nets), 'access': n['access']})
        elif 'id' in n:
            networks_new.append({'id': n['id'], 'access': n['access']})
    return networks_new