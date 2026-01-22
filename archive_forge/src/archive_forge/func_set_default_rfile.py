import os
from shutil import which
from .. import config
from .base import (
def set_default_rfile(self, rfile):
    """Set the default R script file format for R classes.

        This method is used to set values for all R
        subclasses.
        """
    self._rfile = rfile