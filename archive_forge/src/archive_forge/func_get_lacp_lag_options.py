from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def get_lacp_lag_options(self, lag):
    """Get and check LACP LAG options"""
    lag_name = lag.get('name', None)
    if lag_name is None:
        self.module.fail_json(msg="Please specify name in lag options as it's a required parameter")
    lag_mode = lag.get('mode', None)
    if lag_mode is None:
        self.module.fail_json(msg="Please specify mode in lag options as it's a required parameter")
    lag_uplink_number = lag.get('uplink_number', None)
    if lag_uplink_number is None:
        self.module.fail_json(msg="Please specify uplink_number in lag options as it's a required parameter")
    try:
        lag_uplink_number = int(lag_uplink_number)
    except ValueError:
        self.module.fail_json(msg='Failed to parse uplink_number in lag options')
    if lag_uplink_number > 30:
        self.module.fail_json(msg='More than 30 uplinks are not supported in a single LAG!')
    lag_load_balancing_mode = lag.get('load_balancing_mode', None)
    supported_lb_modes = ['srcTcpUdpPort', 'srcDestIpTcpUdpPortVlan', 'srcIpVlan', 'srcDestTcpUdpPort', 'srcMac', 'destIp', 'destMac', 'vlan', 'srcDestIp', 'srcIpTcpUdpPortVlan', 'srcDestIpTcpUdpPort', 'srcDestMac', 'destIpTcpUdpPort', 'srcPortId', 'srcIp', 'srcIpTcpUdpPort', 'destIpTcpUdpPortVlan', 'destTcpUdpPort', 'destIpVlan', 'srcDestIpVlan']
    if lag_load_balancing_mode is None:
        self.module.fail_json(msg="Please specify load_balancing_mode in lag options as it's a required parameter")
    elif lag_load_balancing_mode not in supported_lb_modes:
        self.module.fail_json(msg="The specified load balancing mode '%s' isn't supported!" % lag_load_balancing_mode)
    return (lag_name, lag_mode, lag_uplink_number, lag_load_balancing_mode)