from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def filter_router_isis_data(json):
    option_list = ['adjacency_check', 'adjacency_check6', 'adv_passive_only', 'adv_passive_only6', 'auth_keychain_l1', 'auth_keychain_l2', 'auth_mode_l1', 'auth_mode_l2', 'auth_password_l1', 'auth_password_l2', 'auth_sendonly_l1', 'auth_sendonly_l2', 'default_originate', 'default_originate6', 'dynamic_hostname', 'ignore_lsp_errors', 'is_type', 'isis_interface', 'isis_net', 'lsp_gen_interval_l1', 'lsp_gen_interval_l2', 'lsp_refresh_interval', 'max_lsp_lifetime', 'metric_style', 'overload_bit', 'overload_bit_on_startup', 'overload_bit_suppress', 'redistribute', 'redistribute_l1', 'redistribute_l1_list', 'redistribute_l2', 'redistribute_l2_list', 'redistribute6', 'redistribute6_l1', 'redistribute6_l1_list', 'redistribute6_l2', 'redistribute6_l2_list', 'spf_interval_exp_l1', 'spf_interval_exp_l2', 'summary_address', 'summary_address6']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary