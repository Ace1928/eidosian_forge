import os
import os.path
import posixpath
import stat
from .compat import Collection, Iterable, string_types, unicode
def iter_tree_entries(root, on_error=None, follow_links=None):
    """
	Walks the specified directory for all files and directories.

	*root* (:class:`str`) is the root directory to search.

	*on_error* (:class:`~collections.abc.Callable` or :data:`None`)
	optionally is the error handler for file-system exceptions. It will be
	called with the exception (:exc:`OSError`). Reraise the exception to
	abort the walk. Default is :data:`None` to ignore file-system
	exceptions.

	*follow_links* (:class:`bool` or :data:`None`) optionally is whether
	to walk symbolic links that resolve to directories. Default is
	:data:`None` for :data:`True`.

	Raises :exc:`RecursionError` if recursion is detected.

	Returns an :class:`~collections.abc.Iterable` yielding each file or
	directory entry (:class:`.TreeEntry`) relative to *root*.
	"""
    if on_error is not None and (not callable(on_error)):
        raise TypeError('on_error:{!r} is not callable.'.format(on_error))
    if follow_links is None:
        follow_links = True
    for entry in _iter_tree_entries_next(os.path.abspath(root), '', {}, on_error, follow_links):
        yield entry