import posixpath
import stat
from dulwich.errors import NotTreeError
from dulwich.object_store import tree_lookup_path
from dulwich.objects import SubmoduleEncountered
from ..revision import NULL_REVISION
from .mapping import encode_git_path
def find_last_change_revision(self, path, commit_id):
    if not isinstance(path, bytes):
        raise TypeError(path)
    store = self.store
    while True:
        commit = store[commit_id]
        try:
            target_mode, target_sha = tree_lookup_path(store.__getitem__, commit.tree, path)
        except SubmoduleEncountered as e:
            revid = self.repository.lookup_foreign_revision_id(commit_id)
            revtree = self.repository.revision_tree(revid)
            store = revtree._get_submodule_store(e.path)
            commit_id = e.sha
            path = posixpath.relpath(path, e.path)
        else:
            break
    if path == b'':
        target_mode = stat.S_IFDIR
    if target_mode is None:
        raise AssertionError('sha %r for %r in %r' % (target_sha, path, commit_id))
    while True:
        parent_commits = []
        for parent_id in commit.parents:
            try:
                parent_commit = store[parent_id]
                mode, sha = tree_lookup_path(store.__getitem__, parent_commit.tree, path)
            except (KeyError, NotTreeError):
                continue
            else:
                parent_commits.append(parent_commit)
            if path == b'':
                mode = stat.S_IFDIR
            if mode != target_mode or (not stat.S_ISDIR(target_mode) and sha != target_sha):
                return (store, path, commit.id)
        if parent_commits == []:
            break
        commit = parent_commits[0]
    return (store, path, commit.id)