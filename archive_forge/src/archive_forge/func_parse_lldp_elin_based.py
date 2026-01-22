from __future__ import absolute_import, division, print_function
from copy import deepcopy
from re import M, findall, search
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.argspec.lldp_interfaces.lldp_interfaces import (
def parse_lldp_elin_based(self, conf):
    elin_based = None
    if conf:
        e_num = search('^.* elin (.+)', conf, M)
        elin_based = e_num.group(1).strip("'")
    return elin_based