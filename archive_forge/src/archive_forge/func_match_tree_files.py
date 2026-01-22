from . import util
from .compat import Collection, iterkeys, izip_longest, string_types, unicode
def match_tree_files(self, root, on_error=None, follow_links=None):
    """
		Walks the specified root path for all files and matches them to this
		path-spec.

		*root* (:class:`str`; or :class:`pathlib.PurePath`) is the root
		directory to search for files.

		*on_error* (:class:`~collections.abc.Callable` or :data:`None`)
		optionally is the error handler for file-system exceptions. See
		:func:`~pathspec.util.iter_tree_files` for more information.

		*follow_links* (:class:`bool` or :data:`None`) optionally is whether
		to walk symbolic links that resolve to directories. See
		:func:`~pathspec.util.iter_tree_files` for more information.

		Returns the matched files (:class:`~collections.abc.Iterable` of
		:class:`str`).
		"""
    files = util.iter_tree_files(root, on_error=on_error, follow_links=follow_links)
    return self.match_files(files)