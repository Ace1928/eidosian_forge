from ... import errors, tests, transport
from .. import index as _mod_index
def test_bad_index_value(self):
    error = _mod_index.BadIndexValue('foo')
    self.assertEqual("The value 'foo' is not a valid value.", str(error))