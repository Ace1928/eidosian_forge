from . import util
from .compat import Collection, iterkeys, izip_longest, string_types, unicode
def match_files(self, files, separators=None):
    """
		Matches the files to this path-spec.

		*files* (:class:`~collections.abc.Iterable` of :class:`str; or
		:class:`pathlib.PurePath`) contains the file paths to be matched
		against :attr:`self.patterns <PathSpec.patterns>`.

		*separators* (:class:`~collections.abc.Collection` of :class:`str`;
		or :data:`None`) optionally contains the path separators to
		normalize. See :func:`~pathspec.util.normalize_file` for more
		information.

		Returns the matched files (:class:`~collections.abc.Iterable` of
		:class:`str`).
		"""
    if not util._is_iterable(files):
        raise TypeError('files:{!r} is not an iterable.'.format(files))
    file_map = util.normalize_files(files, separators=separators)
    matched_files = util.match_files(self.patterns, iterkeys(file_map))
    for path in matched_files:
        yield file_map[path]