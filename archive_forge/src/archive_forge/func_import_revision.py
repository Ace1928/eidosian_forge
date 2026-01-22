from ..push import PushResult
from .errors import GitSmartRemoteNotSupported
def import_revision(self, revid, lossy):
    """Import a revision into this Git repository.

        :param revid: Revision id of the revision
        :param roundtrip: Whether to roundtrip bzr metadata
        """
    tree = self._object_store.tree_cache.revision_tree(revid)
    rev = self.source.get_revision(revid)
    commit = None
    for path, obj in self._object_store._revision_to_objects(rev, tree, lossy):
        if obj.type_name == b'commit':
            commit = obj
        self._pending.append((obj, path))
    if commit is None:
        raise AssertionError('no commit object generated for revision %s' % revid)
    return commit.id