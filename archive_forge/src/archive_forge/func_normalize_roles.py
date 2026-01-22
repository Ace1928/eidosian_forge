from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.basic import AnsibleModule, json
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import (
def normalize_roles(meraki, servers):
    if len(servers['servers']) > 0:
        for server in range(len(servers)):
            for role in range(len(servers['servers'][server]['roles'])):
                servers['servers'][server]['roles'][role] = servers['servers'][server]['roles'][role].lower()
    return servers