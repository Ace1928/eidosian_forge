from __future__ import (absolute_import, division, print_function)
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, get_nc_config, set_nc_config
def check_ipaddr_validate(self):
    """Check ipaddress validate"""
    rule1 = '(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])\\.'
    rule2 = '(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])'
    ipv4_regex = '%s%s%s%s%s%s' % ('^', rule1, rule1, rule1, rule2, '$')
    ipv6_regex = '^(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}$'
    flag = False
    if bool(re.match(ipv4_regex, self.address)):
        flag = True
        self.ip_ver = 'IPv4'
        if not self.ntp_ucast_ipv4_validate():
            flag = False
    elif bool(re.match(ipv6_regex, self.address)):
        flag = True
        self.ip_ver = 'IPv6'
    else:
        flag = True
        self.ip_ver = 'IPv6'
    if not flag:
        if self.peer_type == 'Server':
            self.module.fail_json(msg='Error: Illegal server ip-address.')
        else:
            self.module.fail_json(msg='Error: Illegal peer ip-address.')