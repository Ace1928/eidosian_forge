import os
from sys import platform
import shutil
from ... import logging, LooseVersion
from ...utils.filemanip import split_filename, fname_presuffix
from ..base import (
from ...external.due import BibTeX
@classmethod
def set_default_output_type(cls, outputtype):
    """
        Set the default output type for AFNI classes.

        This method is used to set the default output type for all afni
        subclasses.  However, setting this will not update the output
        type for any existing instances.  For these, assign the
        <instance>.inputs.outputtype.
        """
    if outputtype in Info.ftypes:
        cls._outputtype = outputtype
    else:
        raise AttributeError('Invalid AFNI outputtype: %s' % outputtype)