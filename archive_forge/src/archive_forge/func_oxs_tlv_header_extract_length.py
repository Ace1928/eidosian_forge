from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_utils
from os_ken.ofproto import oxm_fields
from os_ken.ofproto import oxs_fields
from struct import calcsize
def oxs_tlv_header_extract_length(header):
    return header & 255