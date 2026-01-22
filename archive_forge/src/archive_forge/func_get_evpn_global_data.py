from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.argspec.evpn_global.evpn_global import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.evpn_global import (
def get_evpn_global_data(self, connection):
    return connection.get('show running-config | section ^l2vpn evpn$')