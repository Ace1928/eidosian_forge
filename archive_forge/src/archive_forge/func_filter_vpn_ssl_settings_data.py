from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def filter_vpn_ssl_settings_data(json):
    option_list = ['algorithm', 'auth_session_check_source_ip', 'auth_timeout', 'authentication_rule', 'auto_tunnel_static_route', 'banned_cipher', 'browser_language_detection', 'check_referer', 'ciphersuite', 'client_sigalgs', 'default_portal', 'deflate_compression_level', 'deflate_min_data_size', 'dns_server1', 'dns_server2', 'dns_suffix', 'dtls_heartbeat_fail_count', 'dtls_heartbeat_idle_timeout', 'dtls_heartbeat_interval', 'dtls_hello_timeout', 'dtls_max_proto_ver', 'dtls_min_proto_ver', 'dtls_tunnel', 'dual_stack_mode', 'encode_2f_sequence', 'encrypt_and_store_password', 'force_two_factor_auth', 'header_x_forwarded_for', 'hsts_include_subdomains', 'http_compression', 'http_only_cookie', 'http_request_body_timeout', 'http_request_header_timeout', 'https_redirect', 'idle_timeout', 'ipv6_dns_server1', 'ipv6_dns_server2', 'ipv6_wins_server1', 'ipv6_wins_server2', 'login_attempt_limit', 'login_block_time', 'login_timeout', 'port', 'port_precedence', 'reqclientcert', 'route_source_interface', 'saml_redirect_port', 'server_hostname', 'servercert', 'source_address', 'source_address_negate', 'source_address6', 'source_address6_negate', 'source_interface', 'ssl_client_renegotiation', 'ssl_insert_empty_fragment', 'ssl_max_proto_ver', 'ssl_min_proto_ver', 'status', 'tlsv1_0', 'tlsv1_1', 'tlsv1_2', 'tlsv1_3', 'transform_backward_slashes', 'tunnel_addr_assigned_method', 'tunnel_connect_without_reauth', 'tunnel_ip_pools', 'tunnel_ipv6_pools', 'tunnel_user_session_timeout', 'unsafe_legacy_renegotiation', 'url_obscuration', 'user_peer', 'web_mode_snat', 'wins_server1', 'wins_server2', 'x_content_type_options', 'ztna_trusted_client']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary