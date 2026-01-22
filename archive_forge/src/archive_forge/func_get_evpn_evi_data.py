from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.argspec.evpn_evi.evpn_evi import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.evpn_evi import (
def get_evpn_evi_data(self, connection):
    return connection.get('show running-config | section ^l2vpn evpn instance .+$')