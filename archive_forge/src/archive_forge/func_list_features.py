import os
import re
import sys
import unittest
from numba.core import config
from numba.misc.gdb_hook import _confirm_gdb
from numba.misc.numba_gdbinfo import collect_gdbinfo
def list_features(self):
    """No equivalent gdb command? Returns a list of supported gdb
           features.
        """
    self._list_features_raw()
    output = self._captured.after
    decoded = output.decode('utf-8')
    m = re.match('.*features=\\[(.*)\\].*', decoded)
    assert m is not None, 'No match found for features string'
    g = m.groups()
    assert len(g) == 1, 'Invalid number of match groups found'
    return g[0].replace('"', '').split(',')