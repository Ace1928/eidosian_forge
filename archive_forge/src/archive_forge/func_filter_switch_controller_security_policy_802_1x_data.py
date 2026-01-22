from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.comparison import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.comparison import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.comparison import (
def filter_switch_controller_security_policy_802_1x_data(json):
    option_list = ['auth_fail_vlan', 'auth_fail_vlan_id', 'auth_fail_vlanid', 'authserver_timeout_period', 'authserver_timeout_vlan', 'authserver_timeout_vlanid', 'eap_auto_untagged_vlans', 'eap_passthru', 'framevid_apply', 'guest_auth_delay', 'guest_vlan', 'guest_vlan_id', 'guest_vlanid', 'mac_auth_bypass', 'name', 'open_auth', 'policy_type', 'radius_timeout_overwrite', 'security_mode', 'user_group']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary