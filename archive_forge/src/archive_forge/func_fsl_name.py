import os
from copy import deepcopy
import pytest
import pdb
from nipype.utils.filemanip import split_filename, ensure_list
from .. import preprocess as fsl
from nipype.interfaces.fsl import Info
from nipype.interfaces.base import File, TraitError, Undefined, isdefined
from nipype.interfaces.fsl import no_fsl
def fsl_name(obj, fname):
    """Create valid fsl name, including file extension for output type."""
    ext = Info.output_type_to_ext(obj.inputs.output_type)
    return fname + ext