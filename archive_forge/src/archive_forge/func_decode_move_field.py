from ovs.flow.decoders import (
from ovs.flow.kv import (
from ovs.flow.list import nested_list_decoder, ListDecoders
from ovs.flow.ofp_fields import field_decoders, field_aliases
def decode_move_field(value):
    """Decodes MOVE actions such as 'move:src->dst'."""
    parts = value.split('->')
    if len(parts) != 2:
        raise ValueError('Malformed move action : %s' % value)
    return {'src': decode_field(parts[0]), 'dst': decode_field(parts[1])}