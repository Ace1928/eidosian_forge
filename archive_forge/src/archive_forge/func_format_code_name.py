import __future__
from ast import PyCF_ONLY_AST
import codeop
import functools
import hashlib
import linecache
import operator
import time
from contextlib import contextmanager
def format_code_name(self, name):
    """Return a user-friendly label and name for a code block.

        Parameters
        ----------
        name : str
            The name for the code block returned from get_code_name

        Returns
        -------
        A (label, name) pair that can be used in tracebacks, or None if the default formatting should be used.
        """
    if name in self._filename_map:
        return ('Cell', 'In[%s]' % self._filename_map[name])