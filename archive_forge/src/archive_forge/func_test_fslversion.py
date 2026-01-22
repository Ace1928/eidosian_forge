import os
import nipype.interfaces.fsl as fsl
from nipype.interfaces.base import InterfaceResult
from nipype.interfaces.fsl import check_fsl, no_fsl
import pytest
@pytest.mark.skipif(no_fsl(), reason='fsl is not installed')
def test_fslversion():
    ver = fsl.Info.version()
    assert ver.split('.', 1)[0].isdigit()