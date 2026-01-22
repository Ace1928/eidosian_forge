from statsmodels.compat.python import lmap, lrange
from itertools import cycle, zip_longest
import csv
def insert_stub(self, loc, stub):
    """Return None.  Inserts a stub cell
        in the row at `loc`.
        """
    _Cell = self._Cell
    if not isinstance(stub, _Cell):
        stub = stub
        stub = _Cell(stub, datatype='stub', row=self)
    self.insert(loc, stub)