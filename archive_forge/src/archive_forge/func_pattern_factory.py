import os
import os.path
import posixpath
import stat
from .compat import Collection, Iterable, string_types, unicode
@property
def pattern_factory(self):
    """
		*pattern_factory* (:class:`~collections.abc.Callable`) is the
		registered pattern factory.
		"""
    return self.args[1]