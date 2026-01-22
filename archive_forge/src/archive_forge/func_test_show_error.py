import unittest
from ... import tests, transport, ui
from ..ui_testing import StringIOAsTTY, StringIOWithEncoding, TextUIFactory
def test_show_error(self):
    msg = 'an error occurred'
    self.factory.show_error(msg)
    self._check_show_error(msg)