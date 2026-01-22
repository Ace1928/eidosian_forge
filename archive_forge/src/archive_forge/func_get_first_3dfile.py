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
def get_first_3dfile(in_files):
    if not func_is_3d(in_files):
        return None
    if isinstance(in_files[0], list):
        return in_files[0]
    return in_files