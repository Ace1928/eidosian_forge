from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, json
from ansible.module_utils.common.dict_transformations import recursive_diff
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import MerakiModule, meraki_argument_spec
def meraki_struct_to_custom_format(data):
    new_struct = {}
    for interface in INT_NAMES:
        if interface in data['bandwidthLimits']:
            new_struct[interface] = {'bandwidth_limits': {'limit_up': data['bandwidthLimits'][interface]['limitUp'], 'limit_down': data['bandwidthLimits'][interface]['limitDown']}}
    return new_struct