import os
from copy import deepcopy
import pytest
import pdb
from nipype.utils.filemanip import split_filename, ensure_list
from .. import preprocess as fsl
from nipype.interfaces.fsl import Info
from nipype.interfaces.base import File, TraitError, Undefined, isdefined
from nipype.interfaces.fsl import no_fsl
@pytest.mark.skipif(no_fsl(), reason='fsl is not installed')
def test_mcflirt_noinput():
    fnt = fsl.MCFLIRT()
    with pytest.raises(ValueError) as excinfo:
        fnt.run()
    assert str(excinfo.value).startswith("MCFLIRT requires a value for input 'in_file'")