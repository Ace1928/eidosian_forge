import os
from copy import deepcopy
import pytest
from .... import config
from ....interfaces import utility as niu
from ....interfaces import base as nib
from ... import engine as pe
from ..utils import merge_dict
from .test_base import EngineTestInterface
from .test_utils import UtilsTestInterface
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
import nipype.interfaces.spm as spm
import os
from io import StringIO
from nipype.utils.config import config
def test_outputmultipath_collapse(tmpdir):
    """Test an OutputMultiPath whose initial value is ``[[x]]`` to ensure that
    it is returned as ``[x]``, regardless of how accessed."""
    select_if = niu.Select(inlist=[[1, 2, 3], [4]], index=1)
    select_nd = pe.Node(niu.Select(inlist=[[1, 2, 3], [4]], index=1), name='select_nd')
    ifres = select_if.run()
    ndres = select_nd.run()
    assert ifres.outputs.out == [4]
    assert ndres.outputs.out == [4]
    assert select_nd.result.outputs.out == [4]