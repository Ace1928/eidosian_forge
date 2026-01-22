from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def filter_web_proxy_explicit_data(json):
    option_list = ['ftp_incoming_port', 'ftp_over_http', 'http_connection_mode', 'http_incoming_port', 'https_incoming_port', 'https_replacement_message', 'incoming_ip', 'incoming_ip6', 'ipv6_status', 'message_upon_server_error', 'outgoing_ip', 'outgoing_ip6', 'pac_file_data', 'pac_file_name', 'pac_file_server_port', 'pac_file_server_status', 'pac_file_through_https', 'pac_file_url', 'pac_policy', 'pref_dns_result', 'realm', 'sec_default_action', 'secure_web_proxy', 'secure_web_proxy_cert', 'socks', 'socks_incoming_port', 'ssl_algorithm', 'ssl_dh_bits', 'status', 'strict_guest', 'trace_auth_no_rsp', 'unknown_http_version']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary