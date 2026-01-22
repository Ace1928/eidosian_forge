import functools
import inspect
import sys
import msgpack
import rapidjson
from ruamel import yaml
def from_msgpack(b, *, max_bin_len=MAX_BIN_LEN, max_str_len=MAX_STR_LEN):
    """
    Convert a msgpack byte array into Python objects (including rpcq objects)
    """
    return msgpack.loads(b, object_hook=_object_hook, raw=False, max_bin_len=max_bin_len, max_str_len=max_str_len, strict_map_key=False)