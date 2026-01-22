import re
from tempfile import TemporaryFile
from breezy.tests import TestCase
from .. import rio
from ..rio import RioReader, Stanza, read_stanza, read_stanzas, rio_file
def test_tricky_quoted(self):
    tmpf = TemporaryFile()
    tmpf.write(b's: "one"\n\ns: \n\t"one"\n\t\n\ns: "\n\ns: ""\n\ns: """\n\ns: \n\t\n\ns: \\\n\ns: \n\t\\\n\t\\\\\n\t\n\ns: word\\\n\ns: quote"\n\ns: backslashes\\\\\\\n\ns: both\\"\n\n')
    tmpf.seek(0)
    expected_vals = ['"one"', '\n"one"\n', '"', '""', '"""', '\n', '\\', '\n\\\n\\\\\n', 'word\\', 'quote"', 'backslashes\\\\\\', 'both\\"']
    for expected in expected_vals:
        stanza = read_stanza(tmpf)
        self.rio_file_stanzas([stanza])
        self.assertEqual(len(stanza), 1)
        self.assertEqual(stanza.get('s'), expected)