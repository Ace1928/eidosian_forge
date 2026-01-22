import re
from tempfile import TemporaryFile
from breezy.tests import TestCase
from .. import rio
from ..rio import RioReader, Stanza, read_stanza, read_stanzas, rio_file
def test_stanza(self):
    """Construct rio stanza in memory"""
    s = Stanza(number='42', name='fred')
    self.assertTrue('number' in s)
    self.assertFalse('color' in s)
    self.assertFalse('42' in s)
    self.assertEqual(list(s.iter_pairs()), [('name', 'fred'), ('number', '42')])
    self.assertEqual(s.get('number'), '42')
    self.assertEqual(s.get('name'), 'fred')