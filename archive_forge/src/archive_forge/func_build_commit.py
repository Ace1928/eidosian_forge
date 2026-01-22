from . import commit, controldir, errors, revision
def build_commit(self, parent_ids=None, allow_leftmost_as_ghost=False, **commit_kwargs):
    """Build a commit on the branch.

        This makes a commit with no real file content for when you only want
        to look at the revision graph structure.

        :param commit_kwargs: Arguments to pass through to commit, such as
             timestamp.
        """
    if parent_ids is not None:
        if len(parent_ids) == 0:
            base_id = revision.NULL_REVISION
        else:
            base_id = parent_ids[0]
        if base_id != self._branch.last_revision():
            self._move_branch_pointer(base_id, allow_leftmost_as_ghost=allow_leftmost_as_ghost)
    tree = self._branch.create_memorytree()
    with tree.lock_write():
        if parent_ids is not None:
            tree.set_parent_ids(parent_ids, allow_leftmost_as_ghost=allow_leftmost_as_ghost)
        tree.add('')
        return self._do_commit(tree, **commit_kwargs)