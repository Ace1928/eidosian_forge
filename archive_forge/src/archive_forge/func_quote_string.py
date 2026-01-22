import copy
import random
import string
from typing import List, Tuple
import redis
from redis.typing import KeysT, KeyT
def quote_string(v):
    """
    RedisGraph strings must be quoted,
    quote_string wraps given v with quotes incase
    v is a string.
    """
    if isinstance(v, bytes):
        v = v.decode()
    elif not isinstance(v, str):
        return v
    if len(v) == 0:
        return '""'
    v = v.replace('\\', '\\\\')
    v = v.replace('"', '\\"')
    return f'"{v}"'