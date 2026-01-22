from . import util
from .compat import Collection, iterkeys, izip_longest, string_types, unicode
def match_entries(self, entries, separators=None):
    """
		Matches the entries to this path-spec.

		*entries* (:class:`~collections.abc.Iterable` of :class:`~util.TreeEntry`)
		contains the entries to be matched against :attr:`self.patterns <PathSpec.patterns>`.

		*separators* (:class:`~collections.abc.Collection` of :class:`str`;
		or :data:`None`) optionally contains the path separators to
		normalize. See :func:`~pathspec.util.normalize_file` for more
		information.

		Returns the matched entries (:class:`~collections.abc.Iterable` of
		:class:`~util.TreeEntry`).
		"""
    if not util._is_iterable(entries):
        raise TypeError('entries:{!r} is not an iterable.'.format(entries))
    entry_map = util._normalize_entries(entries, separators=separators)
    match_paths = util.match_files(self.patterns, iterkeys(entry_map))
    for path in match_paths:
        yield entry_map[path]