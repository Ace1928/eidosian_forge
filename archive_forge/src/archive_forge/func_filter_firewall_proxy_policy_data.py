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
def filter_firewall_proxy_policy_data(json):
    option_list = ['access_proxy', 'access_proxy6', 'action', 'application_list', 'av_profile', 'block_notification', 'casb_profile', 'cifs_profile', 'comments', 'decrypted_traffic_mirror', 'detect_https_in_http_request', 'device_ownership', 'disclaimer', 'dlp_profile', 'dlp_sensor', 'dstaddr', 'dstaddr_negate', 'dstaddr6', 'dstintf', 'emailfilter_profile', 'file_filter_profile', 'global_label', 'groups', 'http_tunnel_auth', 'icap_profile', 'internet_service', 'internet_service_custom', 'internet_service_custom_group', 'internet_service_group', 'internet_service_id', 'internet_service_name', 'internet_service_negate', 'internet_service6', 'internet_service6_custom', 'internet_service6_custom_group', 'internet_service6_group', 'internet_service6_name', 'internet_service6_negate', 'ips_sensor', 'ips_voip_filter', 'label', 'logtraffic', 'logtraffic_start', 'mms_profile', 'name', 'policyid', 'poolname', 'profile_group', 'profile_protocol_options', 'profile_type', 'proxy', 'redirect_url', 'replacemsg_override_group', 'scan_botnet_connections', 'schedule', 'sctp_filter_profile', 'service', 'service_negate', 'session_ttl', 'spamfilter_profile', 'srcaddr', 'srcaddr_negate', 'srcaddr6', 'srcintf', 'ssh_filter_profile', 'ssh_policy_redirect', 'ssl_ssh_profile', 'status', 'transparent', 'users', 'utm_status', 'uuid', 'videofilter_profile', 'virtual_patch_profile', 'voip_profile', 'waf_profile', 'webcache', 'webcache_https', 'webfilter_profile', 'webproxy_forward_server', 'webproxy_profile', 'ztna_ems_tag', 'ztna_tags_match_logic']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary