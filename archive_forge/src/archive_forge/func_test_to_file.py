import re
from tempfile import TemporaryFile
from breezy.tests import TestCase
from .. import rio
from ..rio import RioReader, Stanza, read_stanza, read_stanzas, rio_file
def test_to_file(self):
    """Write rio to file"""
    tmpf = TemporaryFile()
    s = Stanza(a_thing='something with "quotes like \\"this\\""', number='42', name='fred')
    s.write(tmpf)
    tmpf.seek(0)
    self.assertEqual(tmpf.read(), b'a_thing: something with "quotes like \\"this\\""\nname: fred\nnumber: 42\n')