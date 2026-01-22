from ... import errors, tests, transport
from .. import index as _mod_index
def test_bad_index_options(self):
    error = _mod_index.BadIndexOptions('foo')
    self.assertEqual('Could not parse options for index foo.', str(error))