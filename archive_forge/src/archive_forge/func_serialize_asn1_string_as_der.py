from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.common.text.converters import to_bytes
def serialize_asn1_string_as_der(value):
    """ Deserializes an ASN.1 string to a DER encoded byte string. """
    asn1_match = ASN1_STRING_REGEX.match(value)
    if not asn1_match:
        raise ValueError('The ASN.1 serialized string must be in the format [modifier,]type[:value]')
    tag_type = asn1_match.group('tag_type')
    tag_number = asn1_match.group('tag_number')
    tag_class = asn1_match.group('tag_class') or 'C'
    value_type = asn1_match.group('value_type')
    asn1_value = asn1_match.group('value')
    if value_type != 'UTF8':
        raise ValueError('The ASN.1 serialized string is not a known type "{0}", only UTF8 types are supported'.format(value_type))
    b_value = to_bytes(asn1_value, encoding='utf-8', errors='surrogate_or_strict')
    if not tag_type or (tag_type == 'EXPLICIT' and tag_class != 'U'):
        b_value = pack_asn1(TagClass.universal, False, TagNumber.utf8_string, b_value)
    if tag_type:
        tag_class = {'U': TagClass.universal, 'A': TagClass.application, 'P': TagClass.private, 'C': TagClass.context_specific}[tag_class]
        constructed = tag_type == 'EXPLICIT' and tag_class != TagClass.universal
        b_value = pack_asn1(tag_class, constructed, int(tag_number), b_value)
    return b_value