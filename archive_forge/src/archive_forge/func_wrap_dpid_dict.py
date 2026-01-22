import logging
from os_ken.ofproto import ether
from os_ken.ofproto import ofproto_v1_4
from os_ken.ofproto import ofproto_v1_4_parser
from os_ken.lib import ofctl_utils
def wrap_dpid_dict(dp, value, to_user=True):
    if to_user:
        return {str(dp.id): value}
    return {dp.id: value}