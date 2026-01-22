import os
from ... import tests
from ...conflicts import resolve
from ...tests import scenarios
from ...tests.test_conflicts import vary_by_conflicts
from .. import conflicts as bzr_conflicts
def test_stanza_roundtrip(self):
    p = self.conflict
    o = bzr_conflicts.Conflict.factory(**p.as_stanza().as_dict())
    self.assertEqual(o, p)
    self.assertIsInstance(o.path, str)
    if o.file_id is not None:
        self.assertIsInstance(o.file_id, bytes)
    conflict_path = getattr(o, 'conflict_path', None)
    if conflict_path is not None:
        self.assertIsInstance(conflict_path, str)
    conflict_file_id = getattr(o, 'conflict_file_id', None)
    if conflict_file_id is not None:
        self.assertIsInstance(conflict_file_id, bytes)