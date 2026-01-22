from ovs.flow.decoders import (
from ovs.flow.kv import (
from ovs.flow.list import nested_list_decoder, ListDecoders
from ovs.flow.ofp_fields import field_decoders, field_aliases
def learn_field_decoding_kv(key, value):
    """Decodes a key, value pair from the learn action.
        The key must be a decodable field. The value can be either a value
        in the format defined for the field or another field.
        """
    key_field = decode_field(key)
    try:
        return (key, decode_field(value))
    except ParseError:
        return (key, field_decoders.get(key_field.get('field'))(value))