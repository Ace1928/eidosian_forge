from ... import errors, tests, transport
from .. import index as _mod_index
def test_validate_bad_child_index_errors(self):
    trans = self.get_transport()
    trans.put_bytes('name', b'not an index\n')
    idx1 = _mod_index.GraphIndex(trans, 'name', 13)
    idx = _mod_index.CombinedGraphIndex([idx1])
    self.assertRaises(_mod_index.BadIndexFormatSignature, idx.validate)