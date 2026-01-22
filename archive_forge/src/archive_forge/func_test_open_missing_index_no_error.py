from ... import errors, tests, transport
from .. import index as _mod_index
def test_open_missing_index_no_error(self):
    trans = self.get_transport()
    idx1 = _mod_index.GraphIndex(trans, 'missing', 100)
    idx = _mod_index.CombinedGraphIndex([idx1])