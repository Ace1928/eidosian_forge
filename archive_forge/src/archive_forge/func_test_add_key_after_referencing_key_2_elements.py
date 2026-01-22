from ... import errors, tests, transport
from .. import index as _mod_index
def test_add_key_after_referencing_key_2_elements(self):
    builder = _mod_index.GraphIndexBuilder(reference_lists=1, key_elements=2)
    builder.add_node((b'k', b'ey'), b'data', ([(b'reference', b'tokey')],))
    builder.add_node((b'reference', b'tokey'), b'data', ([],))