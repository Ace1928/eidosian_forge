import os
import os.path
import posixpath
import stat
from .compat import Collection, Iterable, string_types, unicode
def is_dir(self, follow_links=None):
    """
		Get whether the entry is a directory.

		*follow_links* (:class:`bool` or :data:`None`) is whether to follow
		symbolic links. If this is :data:`True`, a symlink to a directory
		will result in :data:`True`. Default is :data:`None` for :data:`True`.

		Returns whether the entry is a directory (:class:`bool`).
		"""
    if follow_links is None:
        follow_links = True
    node_stat = self._stat if follow_links else self._lstat
    return stat.S_ISDIR(node_stat.st_mode)