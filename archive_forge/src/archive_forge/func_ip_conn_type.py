from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_text
import re
@property
def ip_conn_type(self):
    return self.type in ('bond', 'bridge', 'dummy', 'ethernet', '802-3-ethernet', 'generic', 'gre', 'infiniband', 'ipip', 'sit', 'team', 'vlan', 'wifi', '802-11-wireless', 'gsm', 'macvlan', 'wireguard', 'vpn', 'loopback')