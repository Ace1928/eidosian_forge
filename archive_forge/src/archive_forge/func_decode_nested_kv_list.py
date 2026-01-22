import functools
import re
from ovs.flow.decoders import decode_default
def decode_nested_kv_list(decoders, value):
    """A key-value decoder that extracts nested key-value pairs and returns
    them in a list of dictionary.

    Args:
        decoders (KVDecoders): The KVDecoders to use.
        value (str): The value string to decode.
    """
    if not value:
        return True
    parser = KVParser(value, decoders)
    parser.parse()
    return [{kv.key: kv.value} for kv in parser.kv()]