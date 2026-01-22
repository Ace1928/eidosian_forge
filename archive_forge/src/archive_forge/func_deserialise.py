import heapq
import threading
from typing import Callable
from .. import errors, lru_cache, osutils, registry, trace
from .static_tuple import StaticTuple, expect_static_tuple
@classmethod
def deserialise(klass, bytes, key, search_key_func=None):
    """Deserialise bytes to an InternalNode, with key key.

        :param bytes: The bytes of the node.
        :param key: The key that the serialised node has.
        :return: An InternalNode instance.
        """
    key = expect_static_tuple(key)
    return _deserialise_internal_node(bytes, key, search_key_func=search_key_func)