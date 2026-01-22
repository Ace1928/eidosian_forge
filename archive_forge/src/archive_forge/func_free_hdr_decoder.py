from ovs.flow.decoders import (
from ovs.flow.kv import (
from ovs.flow.list import nested_list_decoder, ListDecoders
from ovs.flow.ofp_fields import field_decoders, field_aliases
def free_hdr_decoder(free_val):
    if free_val not in ['ethernet', 'mpls', 'mpls_mc', 'nsh']:
        raise ValueError('Malformed encap action. Unkown header: {}'.format(free_val))
    return ('header', free_val)