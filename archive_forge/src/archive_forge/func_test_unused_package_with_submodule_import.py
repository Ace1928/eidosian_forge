from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_unused_package_with_submodule_import(self):
    """
        When a package and its submodule are imported, only report once.
        """
    checker = self.flakes('\n        import fu\n        import fu.bar\n        ', m.UnusedImport)
    error = checker.messages[0]
    assert error.message == '%r imported but unused'
    assert error.message_args == ('fu.bar',)
    assert error.lineno == 5 if self.withDoctest else 3