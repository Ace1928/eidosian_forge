import os
from .. import config
from .base import (
@classmethod
def set_default_mfile(cls, mfile):
    """Set the default MATLAB script file format for MATLAB classes.

        This method is used to set values for all MATLAB
        subclasses.  However, setting this will not update the output
        type for any existing instances.  For these, assign the
        <instance>.inputs.mfile.
        """
    cls._default_mfile = mfile