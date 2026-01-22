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
def filter_user_radius_data(json):
    option_list = ['account_key_cert_field', 'account_key_processing', 'accounting_server', 'acct_all_servers', 'acct_interim_interval', 'all_usergroup', 'auth_type', 'ca_cert', 'call_station_id_type', 'class', 'client_cert', 'delimiter', 'group_override_attr_type', 'h3c_compatibility', 'interface', 'interface_select_method', 'mac_case', 'mac_password_delimiter', 'mac_username_delimiter', 'name', 'nas_id', 'nas_id_type', 'nas_ip', 'password_encoding', 'password_renewal', 'radius_coa', 'radius_port', 'rsso', 'rsso_context_timeout', 'rsso_endpoint_attribute', 'rsso_endpoint_block_attribute', 'rsso_ep_one_ip_only', 'rsso_flush_ip_session', 'rsso_log_flags', 'rsso_log_period', 'rsso_radius_response', 'rsso_radius_server_port', 'rsso_secret', 'rsso_validate_request_secret', 'secondary_secret', 'secondary_server', 'secret', 'server', 'server_identity_check', 'source_ip', 'sso_attribute', 'sso_attribute_key', 'sso_attribute_value_override', 'status_ttl', 'switch_controller_acct_fast_framedip_detect', 'switch_controller_service_type', 'tertiary_secret', 'tertiary_server', 'timeout', 'tls_min_proto_version', 'transport_protocol', 'use_management_vdom', 'username_case_sensitive']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary