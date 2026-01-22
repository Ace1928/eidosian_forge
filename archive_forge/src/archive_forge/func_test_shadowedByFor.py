from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_shadowedByFor(self):
    """
        Test that shadowing a global name with a for loop variable generates a
        warning.
        """
    self.flakes('\n        import fu\n        fu.bar()\n        for fu in ():\n            pass\n        ', m.ImportShadowedByLoopVar)