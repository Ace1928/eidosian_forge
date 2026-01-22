from ... import errors, tests, transport
from .. import index as _mod_index
def test_key_count_empty(self):
    index = self.make_index()
    self.assertEqual(0, index.key_count())