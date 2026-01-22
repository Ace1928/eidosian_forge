from ... import errors, tests, transport
from .. import index as _mod_index
def test_validate_missing_end_line_empty(self):
    index = self.make_index(2)
    trans = self.get_transport()
    content = trans.get_bytes('index')
    trans.put_bytes('index', content[:-1])
    self.assertRaises(_mod_index.BadIndexData, index.validate)