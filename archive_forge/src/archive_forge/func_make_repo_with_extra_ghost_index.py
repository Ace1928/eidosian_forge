from .... import osutils
from .... import revision as _mod_revision
from ....repository import WriteGroup
from ....tests import TestNotApplicable
from ....tests.per_repository import TestCaseWithRepository
from ... import inventory
from ...knitrepo import RepositoryFormatKnit
def make_repo_with_extra_ghost_index(self):
    """Make a corrupt repository.

        It will contain one revision, b'revision-id'.  The knit index will claim
        that it has one parent, 'incorrect-parent', but the revision text will
        claim it has no parents.

        Note: only the *cache* of the knit index is corrupted.  Thus the
        corruption will only last while the repository is locked.  For this
        reason, the returned repo is locked.
        """
    if not isinstance(self.repository_format, RepositoryFormatKnit):
        raise TestNotApplicable("%s isn't a knit format" % self.repository_format)
    repo = self.make_repository('broken')
    with repo.lock_write(), WriteGroup(repo):
        inv = inventory.Inventory(revision_id=b'revision-id')
        inv.root.revision = b'revision-id'
        inv_sha1 = repo.add_inventory(b'revision-id', inv, [])
        if repo.supports_rich_root():
            root_id = inv.root.file_id
            repo.texts.add_lines((root_id, b'revision-id'), [], [])
        revision = _mod_revision.Revision(b'revision-id', committer='jrandom@example.com', timestamp=0, inventory_sha1=inv_sha1, timezone=0, message='message', parent_ids=[])
        lines = repo._serializer.write_revision_to_lines(revision)
        repo.revisions.add_lines((revision.revision_id,), [(b'incorrect-parent',)], lines)
    repo.lock_write()
    self.addCleanup(repo.unlock)
    return repo