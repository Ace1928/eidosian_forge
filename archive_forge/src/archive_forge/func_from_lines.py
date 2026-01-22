from . import util
from .compat import Collection, iterkeys, izip_longest, string_types, unicode
@classmethod
def from_lines(cls, pattern_factory, lines):
    """
		Compiles the pattern lines.

		*pattern_factory* can be either the name of a registered pattern
		factory (:class:`str`), or a :class:`~collections.abc.Callable` used
		to compile patterns. It must accept an uncompiled pattern (:class:`str`)
		and return the compiled pattern (:class:`.Pattern`).

		*lines* (:class:`~collections.abc.Iterable`) yields each uncompiled
		pattern (:class:`str`). This simply has to yield each line so it can
		be a :class:`file` (e.g., from :func:`open` or :class:`io.StringIO`)
		or the result from :meth:`str.splitlines`.

		Returns the :class:`PathSpec` instance.
		"""
    if isinstance(pattern_factory, string_types):
        pattern_factory = util.lookup_pattern(pattern_factory)
    if not callable(pattern_factory):
        raise TypeError('pattern_factory:{!r} is not callable.'.format(pattern_factory))
    if not util._is_iterable(lines):
        raise TypeError('lines:{!r} is not an iterable.'.format(lines))
    lines = [pattern_factory(line) for line in lines if line]
    return cls(lines)