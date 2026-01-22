import copy
import random
import string
from typing import List, Tuple
import redis
from redis.typing import KeysT, KeyT
def parse_to_list(response):
    """Optimistically parse the response to a list."""
    res = []
    if response is None:
        return res
    for item in response:
        try:
            res.append(int(item))
        except ValueError:
            try:
                res.append(float(item))
            except ValueError:
                res.append(nativestr(item))
        except TypeError:
            res.append(None)
    return res