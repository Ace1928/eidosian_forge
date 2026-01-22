from . import commit, controldir, errors, revision
def build_snapshot(self, parent_ids, actions, message=None, timestamp=None, allow_leftmost_as_ghost=False, committer=None, timezone=None, message_callback=None, revision_id=None):
    """Build a commit, shaped in a specific way.

        Most of the actions are self-explanatory.  'flush' is special action to
        break a series of actions into discrete steps so that complex changes
        (such as unversioning a file-id and re-adding it with a different kind)
        can be expressed in a way that will clearly work.

        :param parent_ids: A list of parent_ids to use for the commit.
            It can be None, which indicates to use the last commit.
        :param actions: A list of actions to perform. Supported actions are:
            ('add', ('path', b'file-id', 'kind', b'content' or None))
            ('modify', ('path', b'new-content'))
            ('unversion', 'path')
            ('rename', ('orig-path', 'new-path'))
            ('flush', None)
        :param message: An optional commit message, if not supplied, a default
            commit message will be written.
        :param message_callback: A message callback to use for the commit, as
            per mutabletree.commit.
        :param timestamp: If non-None, set the timestamp of the commit to this
            value.
        :param timezone: An optional timezone for timestamp.
        :param committer: An optional username to use for commit
        :param allow_leftmost_as_ghost: True if the leftmost parent should be
            permitted to be a ghost.
        :param revision_id: The handle for the new commit, can be None
        :return: The revision_id of the new commit
        """
    if parent_ids is not None:
        if len(parent_ids) == 0:
            base_id = revision.NULL_REVISION
        else:
            base_id = parent_ids[0]
        if base_id != self._branch.last_revision():
            self._move_branch_pointer(base_id, allow_leftmost_as_ghost=allow_leftmost_as_ghost)
    if self._tree is not None:
        tree = self._tree
    else:
        tree = self._branch.create_memorytree()
    with tree.lock_write():
        if parent_ids is not None:
            tree.set_parent_ids(parent_ids, allow_leftmost_as_ghost=allow_leftmost_as_ghost)
        pending = _PendingActions()
        for action, info in actions:
            if action == 'add':
                path, file_id, kind, content = info
                if kind == 'directory':
                    pending.to_add_directories.append((path, file_id))
                else:
                    pending.to_add_files.append(path)
                    pending.to_add_file_ids.append(file_id)
                    pending.to_add_kinds.append(kind)
                    if content is not None:
                        pending.new_contents[path] = content
            elif action == 'modify':
                path, content = info
                pending.new_contents[path] = content
            elif action == 'unversion':
                pending.to_unversion_paths.add(info)
            elif action == 'rename':
                from_relpath, to_relpath = info
                pending.to_rename.append((from_relpath, to_relpath))
            elif action == 'flush':
                self._flush_pending(tree, pending)
                pending = _PendingActions()
            else:
                raise ValueError('Unknown build action: "{}"'.format(action))
        self._flush_pending(tree, pending)
        return self._do_commit(tree, message=message, rev_id=revision_id, timestamp=timestamp, timezone=timezone, committer=committer, message_callback=message_callback)