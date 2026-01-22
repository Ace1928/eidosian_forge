import sys
from twisted.internet._glibbase import ensureNotImported
from twisted.trial.unittest import TestCase
def test_ensureWhenNotImportedDontPrevent(self):
    """
        If the specified modules have never been imported, and import
        prevention is not requested, L{ensureNotImported} has no effect.
        """
    modules = {}
    self.patch(sys, 'modules', modules)
    ensureNotImported(['m1', 'm2'], 'A message.')
    self.assertEqual(modules, {})