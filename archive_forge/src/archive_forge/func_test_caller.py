import sys
from .. import plugin as _mod_plugin
from .. import symbol_versioning, tests
from . import features
def test_caller(message, category=None, stacklevel=1):
    caller = sys._getframe(stacklevel)
    reported_file = caller.f_globals['__file__']
    reported_lineno = caller.f_lineno
    self.assertEqual(__file__, reported_file)
    self.assertEqual(self.lineno + 1, reported_lineno)