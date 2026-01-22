import gc
from io import BytesIO
import logging
import os
import os.path as osp
import stat
import uuid
import git
from git.cmd import Git
from git.compat import defenc
from git.config import GitConfigParser, SectionConstraint, cp
from git.exc import (
from git.objects.base import IndexObject, Object
from git.objects.util import TraversableIterableObj
from git.util import (
from .util import (
from typing import Callable, Dict, Mapping, Sequence, TYPE_CHECKING, cast
from typing import Any, Iterator, Union
from git.types import Commit_ish, Literal, PathLike, TBD
@unbare_repo
def move(self, module_path: PathLike, configuration: bool=True, module: bool=True) -> 'Submodule':
    """Move the submodule to a another module path. This involves physically moving
        the repository at our current path, changing the configuration, as well as
        adjusting our index entry accordingly.

        :param module_path: The path to which to move our module in the parent
            repository's working tree, given as repository - relative or absolute path.
            Intermediate directories will be created accordingly. If the path already
            exists, it must be empty. Trailing (back)slashes are removed automatically.
        :param configuration: If True, the configuration will be adjusted to let
            the submodule point to the given path.
        :param module: If True, the repository managed by this submodule
            will be moved as well. If False, we don't move the submodule's checkout,
            which may leave the parent repository in an inconsistent state.
        :return: self
        :raise ValueError: If the module path existed and was not empty, or was a file.
        :note: Currently the method is not atomic, and it could leave the repository
            in an inconsistent state if a sub-step fails for some reason.
        """
    if module + configuration < 1:
        raise ValueError('You must specify to move at least the module or the configuration of the submodule')
    module_checkout_path = self._to_relative_path(self.repo, module_path)
    if module_checkout_path == self.path:
        return self
    module_checkout_abspath = join_path_native(str(self.repo.working_tree_dir), module_checkout_path)
    if osp.isfile(module_checkout_abspath):
        raise ValueError('Cannot move repository onto a file: %s' % module_checkout_abspath)
    index = self.repo.index
    tekey = index.entry_key(module_checkout_path, 0)
    if configuration and tekey in index.entries:
        raise ValueError('Index entry for target path did already exist')
    if module:
        if osp.exists(module_checkout_abspath):
            if len(os.listdir(module_checkout_abspath)):
                raise ValueError('Destination module directory was not empty')
            if osp.islink(module_checkout_abspath):
                os.remove(module_checkout_abspath)
            else:
                os.rmdir(module_checkout_abspath)
        else:
            pass
    cur_path = self.abspath
    renamed_module = False
    if module and osp.exists(cur_path):
        os.renames(cur_path, module_checkout_abspath)
        renamed_module = True
        if osp.isfile(osp.join(module_checkout_abspath, '.git')):
            module_abspath = self._module_abspath(self.repo, self.path, self.name)
            self._write_git_file_and_module_config(module_checkout_abspath, module_abspath)
    previous_sm_path = self.path
    try:
        if configuration:
            try:
                ekey = index.entry_key(self.path, 0)
                entry = index.entries[ekey]
                del index.entries[ekey]
                nentry = git.IndexEntry(entry[:3] + (module_checkout_path,) + entry[4:])
                index.entries[tekey] = nentry
            except KeyError as e:
                raise InvalidGitRepositoryError("Submodule's entry at %r did not exist" % self.path) from e
            with self.config_writer(index=index) as writer:
                writer.set_value('path', module_checkout_path)
                self.path = module_checkout_path
    except Exception:
        if renamed_module:
            os.renames(module_checkout_abspath, cur_path)
        raise
    if previous_sm_path == self.name:
        self.rename(module_checkout_path)
    return self