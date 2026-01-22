from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.community.hrobot.plugins.module_utils.robot import (
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.common.text.converters import to_native, to_text
def normalize_ip(ip, ip_version):
    if ip is None or ip_version is None:
        return ip
    if '/' in ip:
        ip, range = ip.split('/')
    else:
        ip, range = (ip, '')
    ip_addr = to_native(ipaddress.ip_address(to_text(ip)).compressed)
    if range == '':
        range = '32' if ip_version.lower() == 'ipv4' else '128'
    return ip_addr + '/' + range