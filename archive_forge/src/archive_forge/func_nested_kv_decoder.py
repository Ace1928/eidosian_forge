import functools
import re
from ovs.flow.decoders import decode_default
def nested_kv_decoder(decoders=None, is_list=False):
    """Helper function that creates a nested kv decoder with given
    KVDecoders."""
    if is_list:
        return functools.partial(decode_nested_kv_list, decoders)
    return functools.partial(decode_nested_kv, decoders)