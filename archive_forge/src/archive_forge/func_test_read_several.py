import re
from tempfile import TemporaryFile
from breezy.tests import TestCase
from .. import rio
from ..rio import RioReader, Stanza, read_stanza, read_stanzas, rio_file
def test_read_several(self):
    """Read several stanzas from file"""
    tmpf = TemporaryFile()
    tmpf.write(b'version_header: 1\n\nname: foo\nval: 123\n\nname: quoted\naddress:   "Willowglen"\n\t  42 Wallaby Way\n\t  Sydney\n\nname: bar\nval: 129319\n')
    tmpf.seek(0)
    s = read_stanza(tmpf)
    self.assertEqual(s, Stanza(version_header='1'))
    s = read_stanza(tmpf)
    self.assertEqual(s, Stanza(name='foo', val='123'))
    s = read_stanza(tmpf)
    self.assertEqual(s.get('name'), 'quoted')
    self.assertEqual(s.get('address'), '  "Willowglen"\n  42 Wallaby Way\n  Sydney')
    s = read_stanza(tmpf)
    self.assertEqual(s, Stanza(name='bar', val='129319'))
    s = read_stanza(tmpf)
    self.assertEqual(s, None)
    self.check_rio_file(tmpf)