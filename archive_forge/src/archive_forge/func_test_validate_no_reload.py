from ... import errors, tests, transport
from .. import index as _mod_index
def test_validate_no_reload(self):
    idx, reload_counter = self.make_combined_index_with_missing()
    idx._reload_func = None
    self.assertRaises(transport.NoSuchFile, idx.validate)