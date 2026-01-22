from ovs.flow.decoders import (
from ovs.flow.kv import (
from ovs.flow.list import nested_list_decoder, ListDecoders
from ovs.flow.ofp_fields import field_decoders, field_aliases
def decode_chk_pkt_larger(value):
    """Decodes 'check_pkt_larger(pkt_len)->dst' actions."""
    parts = value.split('->')
    if len(parts) != 2:
        raise ValueError('Malformed check_pkt_larger action : %s' % value)
    pkt_len = int(parts[0].strip('()'))
    dst = decode_field(parts[1])
    return {'pkt_len': pkt_len, 'dst': dst}