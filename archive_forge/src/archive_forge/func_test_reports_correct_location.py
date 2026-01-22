import sys
from .. import plugin as _mod_plugin
from .. import symbol_versioning, tests
from . import features
def test_reports_correct_location(self):
    a_feature = features._CompatabilityThunkFeature(symbol_versioning.deprecated_in((2, 1, 0)), 'breezy.tests.test_features', 'a_feature', 'UnicodeFilenameFeature', replacement_module='breezy.tests.features')

    def test_caller(message, category=None, stacklevel=1):
        caller = sys._getframe(stacklevel)
        reported_file = caller.f_globals['__file__']
        reported_lineno = caller.f_lineno
        self.assertEqual(__file__, reported_file)
        self.assertEqual(self.lineno + 1, reported_lineno)
    self.overrideAttr(symbol_versioning, 'warn', test_caller)
    self.lineno = sys._getframe().f_lineno
    self.requireFeature(a_feature)