import os
from copy import deepcopy
from nibabel import load
import numpy as np
from ... import logging
from ...utils import spm_docs as sd
from ..base import (
from ..base.traits_extension import NoDefaultSpecified
from ..matlab import MatlabCommand
from ...external.due import due, Doi, BibTeX
def func_is_3d(in_file):
    """Checks if input functional files are 3d."""
    if isinstance(in_file, list):
        return func_is_3d(in_file[0])
    else:
        img = load(in_file)
        shape = img.shape
        if len(shape) == 3 or (len(shape) == 4 and shape[3] == 1):
            return True
        else:
            return False