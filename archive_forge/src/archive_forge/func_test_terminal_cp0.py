import codecs
import locale
import sys
from typing import Set
from .. import osutils
from . import TestCase
from .ui_testing import BytesIOWithEncoding, StringIOWithEncoding
def test_terminal_cp0(self):
    self.make_wrapped_streams('cp0', 'cp0', 'cp0', user_encoding='latin-1', enable_fake_encodings=False)
    self.assertEqual('latin-1', osutils.get_terminal_encoding())
    self.assertEqual('', sys.stderr.getvalue())