from ... import errors, tests, transport
from .. import index as _mod_index
def test_bad_index_data(self):
    error = _mod_index.BadIndexData('foo')
    self.assertEqual('Error in data for index foo.', str(error))