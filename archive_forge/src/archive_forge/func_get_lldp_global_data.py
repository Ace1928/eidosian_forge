from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.argspec.lldp_global.lldp_global import (
def get_lldp_global_data(self, connection):
    return connection.get('show running-config | section ^lldp')