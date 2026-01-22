import codecs
import locale
import sys
from typing import Set
from .. import osutils
from . import TestCase
from .ui_testing import BytesIOWithEncoding, StringIOWithEncoding
def test_terminal_cp_unknown(self):
    self.make_wrapped_streams('cp-unknown', 'cp-unknown', 'cp-unknown', user_encoding='latin-1', enable_fake_encodings=False)
    self.assertEqual('latin-1', osutils.get_terminal_encoding())
    self.assertEqual('brz: warning: unknown terminal encoding cp-unknown.\n  Using encoding latin-1 instead.\n', sys.stderr.getvalue())