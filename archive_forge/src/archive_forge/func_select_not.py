import collections
from functools import reduce, singledispatch
from typing import (Any, Dict, Iterable, List, Optional,
import numpy as np
from ase.spectrum.dosdata import DOSData, RawDOSData, GridDOSData, Info
from ase.utils.plotting import SimplePlottingAxes
def select_not(self, **info_selection: str) -> 'DOSCollection':
    """Narrow GridDOSCollection to items without specified info

        For example, if ::

          dc = GridDOSCollection([GridDOSData(x, y1,
                                              info={'a': '1', 'b': '1'}),
                                  GridDOSData(x, y2,
                                              info={'a': '2', 'b': '1'})])

        then ::

          dc.select_not(b='2')

        will return an identical object to dc, while ::

          dc.select_not(a='2')

        will return a DOSCollection with only the first item and ::

          dc.select_not(a='1', b='1')

        will return a DOSCollection with only the second item.

        """
    matches = self._select_to_list(self, info_selection, negative=True)
    if len(matches) == 0:
        return type(self)([], energies=self._energies)
    else:
        return type(self)(matches)