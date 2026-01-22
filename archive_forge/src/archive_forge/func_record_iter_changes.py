import stat
from dulwich.index import commit_tree, read_submodule_head
from dulwich.objects import Blob, Commit
from .. import bugtracker
from .. import config as _mod_config
from .. import gpg, osutils
from .. import revision as _mod_revision
from .. import trace
from ..errors import BzrError, RootMissing, UnsupportedOperation
from ..repository import CommitBuilder
from .mapping import encode_git_path, fix_person_identifier, object_mode
from .tree import entry_factory
def record_iter_changes(self, workingtree, basis_revid, iter_changes):
    seen_root = False
    for change in iter_changes:
        if change.kind == (None, None):
            continue
        if change.versioned[0] and (not change.copied):
            file_id = self._mapping.generate_file_id(change.path[0])
        elif change.versioned[1]:
            file_id = self._mapping.generate_file_id(change.path[1])
        else:
            file_id = None
        if change.path[1]:
            parent_id_new = self._mapping.generate_file_id(osutils.dirname(change.path[1]))
        else:
            parent_id_new = None
        if change.kind[1] in ('directory',):
            self._inv_delta.append((change.path[0], change.path[1], file_id, entry_factory[change.kind[1]](file_id, change.name[1], parent_id_new)))
            if change.kind[0] in ('file', 'symlink'):
                self._blobs[encode_git_path(change.path[0])] = None
                self._any_changes = True
            if change.path[1] == '':
                seen_root = True
            continue
        self._any_changes = True
        if change.path[1] is None:
            self._inv_delta.append((change.path[0], change.path[1], file_id, None))
            self._deleted_paths.add(encode_git_path(change.path[0]))
            continue
        try:
            entry_kls = entry_factory[change.kind[1]]
        except KeyError:
            raise KeyError('unknown kind %s' % change.kind[1])
        entry = entry_kls(file_id, change.name[1], parent_id_new)
        if change.kind[1] == 'file':
            entry.executable = change.executable[1]
            blob = Blob()
            f, st = workingtree.get_file_with_stat(change.path[1])
            try:
                blob.data = f.read()
            finally:
                f.close()
            sha = blob.id
            if st is not None:
                entry.text_size = st.st_size
            else:
                entry.text_size = len(blob.data)
            entry.git_sha1 = sha
            self.store.add_object(blob)
        elif change.kind[1] == 'symlink':
            symlink_target = workingtree.get_symlink_target(change.path[1])
            blob = Blob()
            blob.data = encode_git_path(symlink_target)
            self.store.add_object(blob)
            sha = blob.id
            entry.symlink_target = symlink_target
            st = None
        elif change.kind[1] == 'tree-reference':
            sha = read_submodule_head(workingtree.abspath(change.path[1]))
            reference_revision = workingtree.get_reference_revision(change.path[1])
            entry.reference_revision = reference_revision
            st = None
        else:
            raise AssertionError('Unknown kind %r' % change.kind[1])
        mode = object_mode(change.kind[1], change.executable[1])
        self._inv_delta.append((change.path[0], change.path[1], file_id, entry))
        if change.path[0] is not None:
            self._deleted_paths.add(encode_git_path(change.path[0]))
        self._blobs[encode_git_path(change.path[1])] = (mode, sha)
        if st is not None:
            yield (change.path[1], (entry.git_sha1, st))
    if not seen_root and len(self.parents) == 0:
        raise RootMissing()
    if getattr(workingtree, 'basis_tree', False):
        basis_tree = workingtree.basis_tree()
    else:
        if len(self.parents) == 0:
            basis_revid = _mod_revision.NULL_REVISION
        else:
            basis_revid = self.parents[0]
        basis_tree = self.repository.revision_tree(basis_revid)
    for entry in basis_tree._iter_tree_contents(include_trees=False):
        if entry.path in self._blobs:
            continue
        if entry.path in self._deleted_paths:
            continue
        self._blobs[entry.path] = (entry.mode, entry.sha)
    self.new_inventory = None