from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule, json
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import MerakiModule, meraki_argument_spec
def find_admin(meraki, data, email):
    for a in data:
        if a['email'] == email:
            return a
    return None