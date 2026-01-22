from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def filter_wireless_controller_setting_data(json):
    option_list = ['account_id', 'country', 'darrp_optimize', 'darrp_optimize_schedules', 'device_holdoff', 'device_idle', 'device_weight', 'duplicate_ssid', 'fake_ssid_action', 'fapc_compatibility', 'firmware_provision_on_authorization', 'offending_ssid', 'phishing_ssid_detect', 'wfa_compatibility']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary