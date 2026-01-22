from ovs.flow.decoders import (
from ovs.flow.kv import (
from ovs.flow.list import nested_list_decoder, ListDecoders
from ovs.flow.ofp_fields import field_decoders, field_aliases
def decode_set_field(field_decoders, value):
    """Decodes SET_FIELD actions such as: 'set_field:value/mask->dst'.

    The value is decoded by field_decoders which is a KVDecoders instance.
    Args:
        field_decoders(KVDecoders): The KVDecoders to be used to decode the
            field.
    """
    parts = value.split('->')
    if len(parts) != 2:
        raise ValueError('Malformed set_field action : %s' % value)
    val = parts[0]
    dst = parts[1]
    val_result = field_decoders.decode(dst, val)
    return {'value': {val_result[0]: val_result[1]}, 'dst': decode_field(dst)}