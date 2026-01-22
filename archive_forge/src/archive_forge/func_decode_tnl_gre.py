import re
from functools import partial
from ovs.flow.flow import Flow, Section
from ovs.flow.kv import (
from ovs.flow.decoders import (
def decode_tnl_gre(value):
    """
    Decode tnl_push(header(gre())) action.

    It has the following format:

    gre((flags=0x2000,proto=0x6558),key=0x1e241))

    Args:
        value (str): The value to decode.
    """
    return decode_nested_kv(KVDecoders({'flags': decode_int, 'proto': decode_int, 'key': decode_int, 'csum': decode_int, 'seq': decode_int}), value.replace('(', '').replace(')', ''))