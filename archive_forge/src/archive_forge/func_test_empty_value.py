import re
from tempfile import TemporaryFile
from breezy.tests import TestCase
from .. import rio
from ..rio import RioReader, Stanza, read_stanza, read_stanzas, rio_file
def test_empty_value(self):
    """Serialize stanza with empty field"""
    s = Stanza(empty='')
    self.assertEqual(s.to_string(), b'empty: \n')