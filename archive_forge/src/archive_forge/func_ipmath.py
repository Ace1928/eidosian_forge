from __future__ import absolute_import, division, print_function
from functools import partial
from ansible.errors import AnsibleFilterError
from ansible_collections.ansible.utils.plugins.module_utils.common.argspec_validate import (
from ansible_collections.ansible.utils.plugins.plugin_utils.base.ipaddr_utils import _need_netaddr
def ipmath(value, amount):
    try:
        if '/' in value:
            ip = netaddr.IPNetwork(value).ip
        else:
            ip = netaddr.IPAddress(value)
    except (netaddr.AddrFormatError, ValueError):
        msg = 'You must pass a valid IP address; {0} is invalid'.format(value)
        raise AnsibleFilterError(msg)
    if not isinstance(amount, int):
        msg = 'You must pass an integer for arithmetic; {0} is not a valid integer'.format(amount)
        raise AnsibleFilterError(msg)
    return str(ip + amount)