import sys
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy import tests, transform
from breezy.bzr import inventory, remote
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
def create_branch_with_ghost_text(self):
    builder = self.make_branch_builder('ghost')
    builder.build_snapshot(None, [('add', ('', b'root-id', 'directory', None)), ('add', ('a', b'a-file-id', 'file', b'some content\n'))], revision_id=b'A-id')
    b = builder.get_branch()
    old_rt = b.repository.revision_tree(b'A-id')
    new_inv = inventory.mutable_inventory_from_tree(old_rt)
    new_inv.revision_id = b'B-id'
    new_inv.get_entry(b'a-file-id').revision = b'ghost-id'
    new_rev = _mod_revision.Revision(b'B-id', timestamp=time.time(), timezone=0, message='Committing against a ghost', committer='Joe Foo <joe@foo.com>', properties={}, parent_ids=(b'A-id', b'ghost-id'))
    b.lock_write()
    self.addCleanup(b.unlock)
    b.repository.start_write_group()
    b.repository.add_revision(b'B-id', new_rev, new_inv)
    self.disable_commit_write_group_paranoia(b.repository)
    b.repository.commit_write_group()
    return b