from ... import commands, config, errors, lazy_import, option, osutils
import stat
from breezy import (
def upload_tree(self):
    rev_id = self.get_uploaded_revid()
    if rev_id == revision.NULL_REVISION:
        if not self.quiet:
            self.outf.write('No uploaded revision id found, switching to full upload\n')
        self.upload_full_tree()
        return
    if rev_id == self.rev_id:
        if not self.quiet:
            self.outf.write('Remote location already up to date\n')
    from_tree = self.branch.repository.revision_tree(rev_id)
    self.to_transport.ensure_base()
    changes = self.tree.changes_from(from_tree)
    with self.tree.lock_read():
        for change in changes.removed:
            if self.is_ignored(change.path[0]):
                if not self.quiet:
                    self.outf.write('Ignoring %s\n' % change.path[0])
                continue
            if change.kind[0] == 'file':
                self.delete_remote_file(change.path[0])
            elif change.kind[0] == 'directory':
                self.delete_remote_dir_maybe(change.path[0])
            elif change.kind[0] == 'symlink':
                self.delete_remote_file(change.path[0])
            else:
                raise NotImplementedError
        for change in changes.renamed:
            if self.is_ignored(change.path[0]) and self.is_ignored(change.path[1]):
                if not self.quiet:
                    self.outf.write('Ignoring %s\n' % change.path[0])
                    self.outf.write('Ignoring %s\n' % change.path[1])
                continue
            if change.changed_content:
                self.upload_file(change.path[0], change.path[1])
            self.rename_remote(change.path[0], change.path[1])
        self.finish_renames()
        self.finish_deletions()
        for change in changes.kind_changed:
            if self.is_ignored(change.path[1]):
                if not self.quiet:
                    self.outf.write('Ignoring %s\n' % change.path[1])
                continue
            if change.kind[0] in ('file', 'symlink'):
                self.delete_remote_file(change.path[0])
            elif change.kind[0] == 'directory':
                self.delete_remote_dir(change.path[0])
            else:
                raise NotImplementedError
            if change.kind[1] == 'file':
                self.upload_file(change.path[1], change.path[1])
            elif change.kind[1] == 'symlink':
                target = self.tree.get_symlink_target(change.path[1])
                self.upload_symlink(change.path[1], target)
            elif change.kind[1] == 'directory':
                self.make_remote_dir(change.path[1])
            else:
                raise NotImplementedError
        for change in changes.added + changes.copied:
            if self.is_ignored(change.path[1]):
                if not self.quiet:
                    self.outf.write('Ignoring %s\n' % change.path[1])
                continue
            if change.kind[1] == 'file':
                self.upload_file(change.path[1], change.path[1])
            elif change.kind[1] == 'directory':
                self.make_remote_dir(change.path[1])
            elif change.kind[1] == 'symlink':
                target = self.tree.get_symlink_target(change.path[1])
                try:
                    self.upload_symlink(change.path[1], target)
                except errors.TransportNotPossible:
                    if not self.quiet:
                        self.outf.write('Not uploading symlink %s -> %s\n' % (change.path[1], target))
            else:
                raise NotImplementedError
        for change in changes.modified:
            if self.is_ignored(change.path[1]):
                if not self.quiet:
                    self.outf.write('Ignoring %s\n' % change.path[1])
                continue
            if change.kind[1] == 'file':
                self.upload_file(change.path[1], change.path[1])
            elif change.kind[1] == 'symlink':
                target = self.tree.get_symlink_target(change.path[1])
                self.upload_symlink(change.path[1], target)
            else:
                raise NotImplementedError
        self.set_uploaded_revid(self.rev_id)