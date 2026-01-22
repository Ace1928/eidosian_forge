from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def get_ospf_type(afi):
    return 'ospf' if afi == 'ipv4' else 'ospfv3'