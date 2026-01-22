from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import exec_command, load_config
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec
def get_peers_advertise_type(self):
    """get evpn peer address advertise type list"""
    if len(self.config_list) != 2:
        return None
    self.config_list = self.config.split('l2vpn-family evpn')
    get = re.findall('peer ([0-9]+.[0-9]+.[0-9]+.[0-9]+)\\s?as-number\\s?(\\S*)', self.config_list[0])
    if not get:
        return None
    else:
        peers = list()
        for item in get:
            cmd = 'peer %s advertise arp' % item[0]
            exist1 = is_config_exist(self.config_list[1], cmd)
            cmd = 'peer %s advertise irb' % item[0]
            exist2 = is_config_exist(self.config_list[1], cmd)
            if exist1:
                peers.append(dict(peer_address=item[0], as_number=item[1], advertise_type='arp'))
            if exist2:
                peers.append(dict(peer_address=item[0], as_number=item[1], advertise_type='irb'))
        return peers