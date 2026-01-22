from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def filter_system_ha_data(json):
    option_list = ['arps', 'arps_interval', 'authentication', 'cpu_threshold', 'encryption', 'evpn_ttl', 'failover_hold_time', 'ftp_proxy_threshold', 'gratuitous_arps', 'group_id', 'group_name', 'ha_direct', 'ha_eth_type', 'ha_mgmt_interfaces', 'ha_mgmt_status', 'ha_uptime_diff_margin', 'hb_interval', 'hb_interval_in_milliseconds', 'hb_lost_threshold', 'hbdev', 'hc_eth_type', 'hello_holddown', 'http_proxy_threshold', 'imap_proxy_threshold', 'inter_cluster_session_sync', 'key', 'l2ep_eth_type', 'link_failed_signal', 'load_balance_all', 'logical_sn', 'memory_based_failover', 'memory_compatible_mode', 'memory_failover_flip_timeout', 'memory_failover_monitor_period', 'memory_failover_sample_rate', 'memory_failover_threshold', 'memory_threshold', 'mode', 'monitor', 'multicast_ttl', 'nntp_proxy_threshold', 'override', 'override_wait_time', 'password', 'pingserver_failover_threshold', 'pingserver_flip_timeout', 'pingserver_monitor_interface', 'pingserver_secondary_force_reset', 'pingserver_slave_force_reset', 'pop3_proxy_threshold', 'priority', 'route_hold', 'route_ttl', 'route_wait', 'schedule', 'secondary_vcluster', 'session_pickup', 'session_pickup_connectionless', 'session_pickup_delay', 'session_pickup_expectation', 'session_pickup_nat', 'session_sync_dev', 'smtp_proxy_threshold', 'ssd_failover', 'standalone_config_sync', 'standalone_mgmt_vdom', 'sync_config', 'sync_packet_balance', 'unicast_gateway', 'unicast_hb', 'unicast_hb_netmask', 'unicast_hb_peerip', 'unicast_peers', 'unicast_status', 'uninterruptible_primary_wait', 'uninterruptible_upgrade', 'upgrade_mode', 'vcluster', 'vcluster_id', 'vcluster_status', 'vcluster2', 'vdom', 'weight']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary