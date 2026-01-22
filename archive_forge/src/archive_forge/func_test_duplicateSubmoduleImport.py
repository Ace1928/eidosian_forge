from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_duplicateSubmoduleImport(self):
    """
        If a submodule of a package is imported twice, an unused import warning
        and a redefined while unused warning are reported.
        """
    self.flakes('\n        import fu.bar, fu.bar\n        fu.bar\n        ', m.RedefinedWhileUnused)
    self.flakes('\n        import fu.bar\n        import fu.bar\n        fu.bar\n        ', m.RedefinedWhileUnused)