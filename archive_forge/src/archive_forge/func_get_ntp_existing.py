from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def get_ntp_existing(address, peer_type, module):
    peer_dict = {}
    peer_server_list = []
    peer_list = get_ntp_peer(module)
    for peer in peer_list:
        if peer['address'] == address:
            peer_dict.update(peer)
        else:
            peer_server_list.append(peer)
    source_type, source = get_ntp_source(module)
    if source_type is not None and source is not None:
        peer_dict['source_type'] = source_type
        peer_dict['source'] = source
    return (peer_dict, peer_server_list)