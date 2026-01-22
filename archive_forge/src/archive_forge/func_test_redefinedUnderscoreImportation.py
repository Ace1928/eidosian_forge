from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_redefinedUnderscoreImportation(self):
    """
        Test that shadowing an underscore importation raises a warning.
        """
    self.flakes('\n        from .i18n import _\n        def _(): pass\n        ', m.RedefinedWhileUnused)