from ...tests import TestCase
from ..roundtrip import (CommitSupplement, extract_bzr_metadata,
def test_parent_ids(self):
    metadata = CommitSupplement()
    metadata.explicit_parent_ids = (b'foo', b'bar')
    self.assertEqual(b'parent-ids: foo bar\n', generate_roundtripping_metadata(metadata, 'utf-8'))