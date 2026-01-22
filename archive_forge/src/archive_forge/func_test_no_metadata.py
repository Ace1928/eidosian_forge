from ...tests import TestCase
from ..roundtrip import (CommitSupplement, extract_bzr_metadata,
def test_no_metadata(self):
    metadata = CommitSupplement()
    msg = inject_bzr_metadata(b'Foo', metadata, 'utf-8')
    self.assertEqual(b'Foo', msg)