from ovs.flow.decoders import (
from ovs.flow.kv import (
from ovs.flow.list import nested_list_decoder, ListDecoders
from ovs.flow.ofp_fields import field_decoders, field_aliases
def decode_output(value):
    """Decodes the output value.

    Does not support field specification.
    """
    if len(value.split(',')) > 1:
        return nested_kv_decoder(KVDecoders({'port': decode_default, 'max_len': decode_int}))(value)
    try:
        return {'port': int(value)}
    except ValueError:
        return {'port': value.strip('"')}