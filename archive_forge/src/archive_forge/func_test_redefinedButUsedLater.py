from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_redefinedButUsedLater(self):
    """
        Test that a global import which is redefined locally,
        but used later in another scope does not generate a warning.
        """
    self.flakes("\n        import unittest, transport\n\n        class GetTransportTestCase(unittest.TestCase):\n            def test_get_transport(self):\n                transport = 'transport'\n                self.assertIsNotNone(transport)\n\n        class TestTransportMethodArgs(unittest.TestCase):\n            def test_send_defaults(self):\n                transport.Transport()\n        ")