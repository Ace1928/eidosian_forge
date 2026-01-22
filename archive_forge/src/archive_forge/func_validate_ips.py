from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.facts.facts import Facts
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.l3_interfaces import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.utils.utils import (
def validate_ips(self, afi, want=None, have=None):
    if afi == 'ipv4' and want:
        v4_addr = validate_n_expand_ipv4(self._module, want) if want.get('address') else {}
        if v4_addr:
            want['address'] = v4_addr
    elif afi == 'ipv6' and want:
        if want.get('address'):
            validate_ipv6(want['address'], self._module)
    if afi == 'ipv4' and have:
        v4_addr_h = validate_n_expand_ipv4(self._module, have) if have.get('address') else {}
        if v4_addr_h:
            have['address'] = v4_addr_h
    elif afi == 'ipv6' and have:
        if have.get('address'):
            validate_ipv6(have['address'], self._module)