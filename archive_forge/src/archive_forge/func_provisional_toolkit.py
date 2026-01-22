import sys
import os
from os import path
from contextlib import contextmanager
@contextmanager
def provisional_toolkit(self, toolkit):
    """ Perform an operation with toolkit provisionally set

        This sets the toolkit attribute of the ETSConfig object to the
        provided value. If the operation fails with an exception, the toolkit
        is reset to nothing.

        This method should only be called if the toolkit is not currently set.

        Parameters
        ----------
        toolkit : string
            The name of the toolkit to provisionally use.

        Raises
        ------
        ETSToolkitError
            If the toolkit attribute is already set, then an ETSToolkitError
            will be raised when entering the context manager.
        """
    if self.toolkit:
        msg = "ETSConfig toolkit is already set to '{0}'"
        raise ETSToolkitError(msg.format(self.toolkit))
    self.toolkit = toolkit
    try:
        yield
    except:
        self._toolkit = ''
        raise