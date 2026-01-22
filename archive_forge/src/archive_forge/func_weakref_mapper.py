import types
from collections import namedtuple
from copy import deepcopy
from weakref import ref as _weakref_ref
@staticmethod
def weakref_mapper(encode, val):
    """__autoslot_mappers__ mapper for fields that contain weakrefs

        This mapper expects to be passed a field containing either a
        weakref or None.  It will resolve the weakref to a hard
        reference when generating a state, and then convert the hard
        reference back to a weakref when restoring the state.

        """
    if val is None:
        return val
    if encode:
        return val()
    else:
        return _weakref_ref(val)