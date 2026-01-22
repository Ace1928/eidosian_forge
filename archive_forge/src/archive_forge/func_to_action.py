import logging
import netaddr
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_2_parser
from os_ken.lib import ofctl_utils
def to_action(dp, dic):
    ofp = dp.ofproto
    parser = dp.ofproto_parser
    action_type = dic.get('type')
    return ofctl_utils.to_action(dic, ofp, parser, action_type, UTIL)