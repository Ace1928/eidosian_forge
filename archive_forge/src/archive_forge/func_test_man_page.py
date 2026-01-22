from io import StringIO
import breezy.commands
from . import TestCase
def test_man_page(self):
    from breezy.doc_generate import autodoc_man
    autodoc_man.infogen(self.options, self.sio)
    self.assertNotEqual('', self.sio.getvalue())