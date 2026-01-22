from io import BytesIO
from pprint import pformat
from ... import errors
from ...osutils import sha_string
from ..weave import Weave, WeaveFormatError, WeaveInvalidChecksum
from ..weavefile import read_weave, write_weave
from . import TestCase, TestCaseInTempDir
def test_multi_line_merge(self):
    rawtexts = [b'A Book of Verses underneath the Bough,\n            A Jug of Wine, a Loaf of Bread, -- and Thou\n            Beside me singing in the Wilderness --\n            Oh, Wilderness were Paradise enow!', b'A Book of Verses underneath the Bough,\n            A Jug of Wine, a Loaf of Bread, -- and Thou\n            Beside me singing in the Wilderness --\n            Oh, Wilderness were Paradise now!', b'A Book of poems underneath the tree,\n            A Jug of Wine, a Loaf of Bread,\n            and Thou\n            Beside me singing in the Wilderness --\n            Oh, Wilderness were Paradise now!\n\n            -- O. Khayyam', b'A Book of Verses underneath the Bough,\n            A Jug of Wine, a Loaf of Bread,\n            and Thou\n            Beside me singing in the Wilderness --\n            Oh, Wilderness were Paradise now!']
    texts = [[l.strip() for l in t.split(b'\n')] for t in rawtexts]
    k = Weave()
    parents = set()
    i = 0
    for t in texts:
        k.add_lines(b'text%d' % i, list(parents), t)
        parents.add(b'text%d' % i)
        i += 1
    self.log('k._weave=' + pformat(k._weave))
    for i, t in enumerate(texts):
        self.assertEqual(k.get_lines(i), t)
    self.check_read_write(k)