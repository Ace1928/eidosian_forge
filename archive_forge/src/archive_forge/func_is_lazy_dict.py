import copy
import numbers
from collections.abc import MutableMapping
from warnings import warn
import numpy as np
from nibabel.affines import apply_affine
from .array_sequence import ArraySequence
def is_lazy_dict(obj):
    """True if `obj` seems to implement the :class:`LazyDict` API"""
    return is_data_dict(obj) and callable(list(obj.store.values())[0])