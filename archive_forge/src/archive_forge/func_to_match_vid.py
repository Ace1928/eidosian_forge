import logging
import netaddr
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_2_parser
from os_ken.lib import ofctl_utils
def to_match_vid(value):
    return ofctl_utils.to_match_vid(value, ofproto_v1_2.OFPVID_PRESENT)