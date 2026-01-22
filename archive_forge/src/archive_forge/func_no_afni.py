import os
from sys import platform
import shutil
from ... import logging, LooseVersion
from ...utils.filemanip import split_filename, fname_presuffix
from ..base import (
from ...external.due import BibTeX
def no_afni():
    """Check whether AFNI is not available."""
    if Info.version() is None:
        return True
    return False