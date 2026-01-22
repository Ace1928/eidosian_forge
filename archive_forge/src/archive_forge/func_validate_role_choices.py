from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.basic import AnsibleModule, json
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import (
def validate_role_choices(meraki, servers):
    choices = ['Wireless event log', 'Appliance event log', 'Switch event log', 'Air Marshal events', 'Flows', 'URLs', 'IDS alerts', 'Security events']
    for i in range(len(choices)):
        choices[i] = choices[i].lower()
    for server in range(len(servers)):
        for role in servers[server]['roles']:
            if role.lower() not in choices:
                meraki.fail_json(msg='Invalid role found in {0}.'.format(servers[server]['host']))