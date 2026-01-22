from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ..module_utils.api import WapiModule
from ..module_utils.api import NIOS_HOST_RECORD
from ..module_utils.api import normalize_ib_spec
def ipv6addrs(module):
    return ipaddr(module, 'ipv6addrs', filtered_keys=['address', 'dhcp'])