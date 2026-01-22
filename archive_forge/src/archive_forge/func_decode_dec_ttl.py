from ovs.flow.decoders import (
from ovs.flow.kv import (
from ovs.flow.list import nested_list_decoder, ListDecoders
from ovs.flow.ofp_fields import field_decoders, field_aliases
def decode_dec_ttl(value):
    """Decodes dec_ttl and dec_ttl(id, id[2], ...) actions."""
    if not value:
        return True
    return [int(idx) for idx in value.split(',')]