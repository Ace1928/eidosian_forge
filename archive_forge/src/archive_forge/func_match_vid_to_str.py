import logging
import netaddr
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_2_parser
from os_ken.lib import ofctl_utils
def match_vid_to_str(value, mask):
    return ofctl_utils.match_vid_to_str(value, mask, ofproto_v1_2.OFPVID_PRESENT)