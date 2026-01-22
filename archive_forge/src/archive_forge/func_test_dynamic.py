from dill.source import getsource, getname, _wrap, likely_import
from dill.source import getimportable
from dill._dill import IS_PYPY
import sys
def test_dynamic():
    assert likely_import(add) == 'from %s import add\n' % __name__
    assert likely_import(squared) == 'from %s import squared\n' % __name__