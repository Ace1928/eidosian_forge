from dill.source import getsource, getname, _wrap, likely_import
from dill.source import getimportable
from dill._dill import IS_PYPY
import sys
def test_imported():
    from math import sin
    assert likely_import(sin) == 'from math import sin\n'