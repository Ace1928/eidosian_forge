from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def filter_system_ike_data(json):
    option_list = ['dh_group_1', 'dh_group_14', 'dh_group_15', 'dh_group_16', 'dh_group_17', 'dh_group_18', 'dh_group_19', 'dh_group_2', 'dh_group_20', 'dh_group_21', 'dh_group_27', 'dh_group_28', 'dh_group_29', 'dh_group_30', 'dh_group_31', 'dh_group_32', 'dh_group_5', 'dh_keypair_cache', 'dh_keypair_count', 'dh_keypair_throttle', 'dh_mode', 'dh_multiprocess', 'dh_worker_count', 'embryonic_limit']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary