import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def is_not_container(x):
    """The default function to determine if an object is a leaf. This simply
    checks if the object is an instance of any of the registered container
    types.
    """
    try:
        return IS_CONTAINER_CACHE[x.__class__]
    except KeyError:
        isleaf = not any((isinstance(x, cls) for cls in TREE_MAP_REGISTRY))
        IS_CONTAINER_CACHE[x.__class__] = isleaf
        return isleaf