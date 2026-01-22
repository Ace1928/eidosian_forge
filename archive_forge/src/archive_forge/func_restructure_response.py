from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule, json
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import MerakiModule, meraki_argument_spec
def restructure_response(rules):
    for rule in rules['rules']:
        type = rule['type']
        rule[type] = copy.deepcopy(rule['value'])
        del rule['value']
    return rules