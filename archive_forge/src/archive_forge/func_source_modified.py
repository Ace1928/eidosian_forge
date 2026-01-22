from __future__ import annotations
from . import exc
from . import util as orm_util
from .base import PassiveFlag
def source_modified(uowcommit, source, source_mapper, synchronize_pairs):
    """return true if the source object has changes from an old to a
    new value on the given synchronize pairs

    """
    for l, r in synchronize_pairs:
        try:
            prop = source_mapper._columntoproperty[l]
        except exc.UnmappedColumnError as err:
            _raise_col_to_prop(False, source_mapper, l, None, r, err)
        history = uowcommit.get_attribute_history(source, prop.key, PassiveFlag.PASSIVE_NO_INITIALIZE)
        if bool(history.deleted):
            return True
    else:
        return False