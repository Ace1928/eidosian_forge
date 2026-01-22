from ....revision import Revision
from ....tests import TestCase
from ..pseudonyms import extract_foreign_revids
def test_no_foreign_revid(self):
    x = Revision(b'myrevid')
    self.assertEqual(set(), extract_foreign_revids(x))