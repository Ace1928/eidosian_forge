from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, json
from ansible.module_utils.common.dict_transformations import recursive_diff
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import MerakiModule, meraki_argument_spec
def list_int_to_str(data):
    return [str(item) for item in data]