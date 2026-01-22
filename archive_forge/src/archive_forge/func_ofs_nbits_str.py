import sys
from struct import calcsize
from os_ken.lib import type_desc
from os_ken.ofproto.ofproto_common import OFP_HEADER_SIZE
from os_ken.ofproto import oxm_fields
def ofs_nbits_str(n):
    start = 0
    while True:
        start += 1
        if start << 6 > n:
            break
    start -= 1
    end = n + start - (start << 6)
    return '[%d..%d]' % (start, end)