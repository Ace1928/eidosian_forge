from statsmodels.compat.python import lmap, lrange
from itertools import cycle, zip_longest
import csv
def insert_stubs(self, loc, stubs):
    """Return None.  Insert column of stubs at column `loc`.
        If there is a header row, it gets an empty cell.
        So ``len(stubs)`` should equal the number of non-header rows.
        """
    _Cell = self._Cell
    stubs = iter(stubs)
    for row in self:
        if row.datatype == 'header':
            empty_cell = _Cell('', datatype='empty')
            row.insert(loc, empty_cell)
        else:
            try:
                row.insert_stub(loc, next(stubs))
            except StopIteration:
                raise ValueError('length of stubs must match table length')