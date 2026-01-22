from ovs.flow.decoders import (
from ovs.flow.kv import (
from ovs.flow.list import nested_list_decoder, ListDecoders
from ovs.flow.ofp_fields import field_decoders, field_aliases
def decode_bundle(value, load=False):
    """Decode bundle action."""
    result = {}
    keys = ['fields', 'basis', 'algorithm', 'ofport']
    if load:
        keys.append('dst')
    for key in keys:
        parts = value.partition(',')
        nvalue = parts[0]
        value = parts[2]
        if key == 'ofport':
            continue
        result[key] = decode_default(nvalue)
    mvalues = value.split('members:')
    result['members'] = [int(port) for port in mvalues[1].split(',')]
    return result