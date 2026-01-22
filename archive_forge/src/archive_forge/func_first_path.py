import os
import os.path
import posixpath
import stat
from .compat import Collection, Iterable, string_types, unicode
@property
def first_path(self):
    """
		*first_path* (:class:`str`) is the first path encountered for
		:attr:`self.real_path <RecursionError.real_path>`.
		"""
    return self.args[1]