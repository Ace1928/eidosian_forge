import types
from collections import namedtuple
from copy import deepcopy
from weakref import ref as _weakref_ref
@staticmethod
def encode_as_none(encode, val):
    """__autoslot_mappers__ mapper that will replace fields with None

        This mapper will encode the field as None (regardless of the
        current field value).  No mapping occurs when restoring a state.

        """
    if encode:
        return None
    else:
        return val