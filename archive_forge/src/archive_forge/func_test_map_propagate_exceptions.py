import importlib.metadata as importlib_metadata
import operator
from unittest import mock
from stevedore import exception
from stevedore import extension
from stevedore.tests import utils
def test_map_propagate_exceptions(self):

    def mapped(ext, *args, **kwds):
        raise RuntimeError('hard coded error')
    em = extension.ExtensionManager('stevedore.test.extension', invoke_on_load=True, propagate_map_exceptions=True)
    try:
        em.map(mapped, 1, 2, a='A', b='B')
        assert False
    except RuntimeError:
        pass