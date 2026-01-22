from ... import tests
from .. import rio
def test_non_ascii_char(self):
    self.assertReadStanza(rio.Stanza(foo='nåme'), ['foo: nåme\n'.encode()])