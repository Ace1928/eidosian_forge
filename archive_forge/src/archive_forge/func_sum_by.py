import collections
from functools import reduce, singledispatch
from typing import (Any, Dict, Iterable, List, Optional,
import numpy as np
from ase.spectrum.dosdata import DOSData, RawDOSData, GridDOSData, Info
from ase.utils.plotting import SimplePlottingAxes
def sum_by(self, *info_keys: str) -> 'DOSCollection':
    """Return a DOSCollection with some data summed by common attributes

        For example, if ::

          dc = DOSCollection([DOSData(x1, y1, info={'a': '1', 'b': '1'}),
                              DOSData(x2, y2, info={'a': '2', 'b': '1'}),
                              DOSData(x3, y3, info={'a': '2', 'b': '2'})])

        then ::

          dc.sum_by('b')

        will return a collection equivalent to ::

          DOSCollection([DOSData(x1, y1, info={'a': '1', 'b': '1'})
                         + DOSData(x2, y2, info={'a': '2', 'b': '1'}),
                         DOSData(x3, y3, info={'a': '2', 'b': '2'})])

        where the resulting contained DOSData have info attributes of
        {'b': '1'} and {'b': '2'} respectively.

        dc.sum_by('a', 'b') on the other hand would return the full three-entry
        collection, as none of the entries have common 'a' *and* 'b' info.

        """

    def _matching_info_tuples(data: DOSData):
        """Get relevent dict entries in tuple form

            e.g. if data.info = {'a': 1, 'b': 2, 'c': 3}
                 and info_keys = ('a', 'c')

                 then return (('a', 1), ('c': 3))
            """
        matched_keys = set(info_keys) & set(data.info)
        return tuple(sorted([(key, data.info[key]) for key in matched_keys]))
    all_combos = map(_matching_info_tuples, self)
    unique_combos = sorted(set(all_combos))
    collection_data = [self.select(**dict(combo)).sum_all() for combo in unique_combos]
    return type(self)(collection_data)