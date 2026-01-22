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
@classmethod
def set_mlab_paths(cls, matlab_cmd=None, paths=None, use_mcr=None):
    cls._matlab_cmd = matlab_cmd
    cls._paths = paths
    cls._use_mcr = use_mcr
    info_dict = Info.getinfo(matlab_cmd=matlab_cmd, paths=paths, use_mcr=use_mcr)