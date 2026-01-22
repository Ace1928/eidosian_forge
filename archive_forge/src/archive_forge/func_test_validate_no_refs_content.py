from ... import errors, tests, transport
from .. import index as _mod_index
def test_validate_no_refs_content(self):
    index = self.make_index(nodes=[((b'key',), b'value')])
    index.validate()