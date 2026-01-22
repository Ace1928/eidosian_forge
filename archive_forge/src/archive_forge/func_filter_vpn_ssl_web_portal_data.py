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
def filter_vpn_ssl_web_portal_data(json):
    option_list = ['allow_user_access', 'auto_connect', 'bookmark_group', 'client_src_range', 'clipboard', 'custom_lang', 'customize_forticlient_download_url', 'default_protocol', 'default_window_height', 'default_window_width', 'dhcp_ip_overlap', 'dhcp_ra_giaddr', 'dhcp6_ra_linkaddr', 'display_bookmark', 'display_connection_tools', 'display_history', 'display_status', 'dns_server1', 'dns_server2', 'dns_suffix', 'exclusive_routing', 'focus_bookmark', 'forticlient_download', 'forticlient_download_method', 'heading', 'hide_sso_credential', 'host_check', 'host_check_interval', 'host_check_policy', 'ip_mode', 'ip_pools', 'ipv6_dns_server1', 'ipv6_dns_server2', 'ipv6_exclusive_routing', 'ipv6_pools', 'ipv6_service_restriction', 'ipv6_split_tunneling', 'ipv6_split_tunneling_routing_address', 'ipv6_split_tunneling_routing_negate', 'ipv6_tunnel_mode', 'ipv6_wins_server1', 'ipv6_wins_server2', 'keep_alive', 'landing_page', 'landing_page_mode', 'limit_user_logins', 'mac_addr_action', 'mac_addr_check', 'mac_addr_check_rule', 'macos_forticlient_download_url', 'name', 'os_check', 'os_check_list', 'prefer_ipv6_dns', 'redir_url', 'rewrite_ip_uri_ui', 'save_password', 'service_restriction', 'skip_check_for_browser', 'skip_check_for_unsupported_browser', 'skip_check_for_unsupported_os', 'smb_max_version', 'smb_min_version', 'smb_ntlmv1_auth', 'smbv1', 'split_dns', 'split_tunneling', 'split_tunneling_routing_address', 'split_tunneling_routing_negate', 'theme', 'transform_backward_slashes', 'tunnel_mode', 'use_sdwan', 'user_bookmark', 'user_group_bookmark', 'web_mode', 'windows_forticlient_download_url', 'wins_server1', 'wins_server2']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary