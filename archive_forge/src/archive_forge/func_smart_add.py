import errno
import itertools
import os
import posixpath
import re
import stat
import sys
from collections import defaultdict
from dulwich.config import ConfigFile as GitConfigFile
from dulwich.file import FileLocked, GitFile
from dulwich.ignore import IgnoreFilterManager
from dulwich.index import (ConflictedIndexEntry, Index, IndexEntry, SHA1Writer,
from dulwich.object_store import iter_tree_contents
from dulwich.objects import S_ISGITLINK
from .. import branch as _mod_branch
from .. import conflicts as _mod_conflicts
from .. import controldir as _mod_controldir
from .. import errors, globbing, lock, osutils
from .. import revision as _mod_revision
from .. import trace
from .. import transport as _mod_transport
from .. import tree, urlutils, workingtree
from ..decorators import only_raises
from ..mutabletree import BadReferenceTarget, MutableTree
from .dir import BareLocalGitControlDirFormat, LocalGitDir
from .mapping import decode_git_path, encode_git_path, mode_kind
from .tree import MutableGitIndexTree
def smart_add(self, file_list, recurse=True, action=None, save=True):
    if not file_list:
        file_list = ['.']
    if self.supports_symlinks():
        file_list = list(map(osutils.normalizepath, file_list))
    conflicts_related = set()
    for c in self.conflicts():
        conflicts_related.update(c.associated_filenames())
    added = []
    ignored = {}
    user_dirs = []

    def call_action(filepath, kind):
        if filepath == '':
            return
        if action is not None:
            parent_path = posixpath.dirname(filepath)
            parent_id = self.path2id(parent_path)
            parent_ie = self._get_dir_ie(parent_path, parent_id)
            file_id = action(self, parent_ie, filepath, kind)
            if file_id is not None:
                raise workingtree.SettingFileIdUnsupported()
    with self.lock_tree_write():
        for filepath in osutils.canonical_relpaths(self.basedir, file_list):
            filepath, can_access = osutils.normalized_filename(filepath)
            if not can_access:
                raise errors.InvalidNormalization(filepath)
            abspath = self.abspath(filepath)
            kind = osutils.file_kind(abspath)
            if kind in ('file', 'symlink'):
                index, subpath = self._lookup_index(encode_git_path(filepath))
                if subpath in index:
                    continue
                call_action(filepath, kind)
                if save:
                    self._index_add_entry(filepath, kind)
                added.append(filepath)
            elif kind == 'directory':
                index, subpath = self._lookup_index(encode_git_path(filepath))
                if subpath not in index:
                    call_action(filepath, kind)
                if recurse:
                    user_dirs.append(filepath)
            else:
                raise errors.BadFileKindError(filename=abspath, kind=kind)
        for user_dir in user_dirs:
            abs_user_dir = self.abspath(user_dir)
            if user_dir != '':
                try:
                    transport = _mod_transport.get_transport_from_path(abs_user_dir)
                    _mod_controldir.ControlDirFormat.find_format(transport)
                    subtree = True
                except errors.NotBranchError:
                    subtree = False
                except errors.UnsupportedFormatError:
                    subtree = False
            else:
                subtree = False
            if subtree:
                trace.warning('skipping nested tree %r', abs_user_dir)
                continue
            for name in os.listdir(abs_user_dir):
                subp = os.path.join(user_dir, name)
                if self.is_control_filename(subp) or self.mapping.is_special_file(subp):
                    continue
                ignore_glob = self.is_ignored(subp)
                if ignore_glob is not None:
                    ignored.setdefault(ignore_glob, []).append(subp)
                    continue
                abspath = self.abspath(subp)
                kind = osutils.file_kind(abspath)
                if kind == 'directory':
                    user_dirs.append(subp)
                else:
                    index, subpath = self._lookup_index(encode_git_path(subp))
                    if subpath in index:
                        continue
                    if subp in conflicts_related:
                        continue
                    call_action(subp, kind)
                    if save:
                        self._index_add_entry(subp, kind)
                    added.append(subp)
        return (added, ignored)