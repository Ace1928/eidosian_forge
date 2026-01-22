from __future__ import annotations
from . import exc
from . import util as orm_util
from .base import PassiveFlag
def populate_dict(source, source_mapper, dict_, synchronize_pairs):
    for l, r in synchronize_pairs:
        try:
            value = source_mapper._get_state_attr_by_column(source, source.dict, l, passive=PassiveFlag.PASSIVE_OFF)
        except exc.UnmappedColumnError as err:
            _raise_col_to_prop(False, source_mapper, l, None, r, err)
        dict_[r.key] = value