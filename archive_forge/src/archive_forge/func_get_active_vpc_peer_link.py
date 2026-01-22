from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def get_active_vpc_peer_link(module):
    peer_link = None
    try:
        body = run_commands(module, ['show vpc brief | json'])[0]
        peer_link = body['TABLE_peerlink']['ROW_peerlink']['peerlink-ifindex']
    except (KeyError, AttributeError, TypeError):
        return peer_link
    return peer_link