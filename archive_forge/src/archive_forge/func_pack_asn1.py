from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.common.text.converters import to_bytes
def pack_asn1(tag_class, constructed, tag_number, b_data):
    """Pack the value into an ASN.1 data structure.

    The structure for an ASN.1 element is

    | Identifier Octet(s) | Length Octet(s) | Data Octet(s) |
    """
    b_asn1_data = bytearray()
    if tag_class < 0 or tag_class > 3:
        raise ValueError('tag_class must be between 0 and 3 not %s' % tag_class)
    identifier_octets = tag_class << 6
    identifier_octets |= (1 if constructed else 0) << 5
    if tag_number < 31:
        identifier_octets |= tag_number
        b_asn1_data.append(identifier_octets)
    else:
        identifier_octets |= 31
        b_asn1_data.append(identifier_octets)
        b_asn1_data.extend(_pack_octet_integer(tag_number))
    length = len(b_data)
    if length < 128:
        b_asn1_data.append(length)
    else:
        length_octets = bytearray()
        while length:
            length_octets.append(length & 255)
            length >>= 8
        length_octets.reverse()
        b_asn1_data.append(len(length_octets) | 128)
        b_asn1_data.extend(length_octets)
    return bytes(b_asn1_data) + b_data