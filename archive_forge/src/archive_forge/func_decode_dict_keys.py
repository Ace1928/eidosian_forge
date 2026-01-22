import copy
import random
import string
from typing import List, Tuple
import redis
from redis.typing import KeysT, KeyT
def decode_dict_keys(obj):
    """Decode the keys of the given dictionary with utf-8."""
    newobj = copy.copy(obj)
    for k in obj.keys():
        if isinstance(k, bytes):
            newobj[k.decode('utf-8')] = newobj[k]
            newobj.pop(k)
    return newobj