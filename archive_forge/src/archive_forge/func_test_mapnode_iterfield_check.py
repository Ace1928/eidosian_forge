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
def test_mapnode_iterfield_check():
    mod1 = pe.MapNode(EngineTestInterface(), iterfield=['input1'], name='mod1')
    with pytest.raises(ValueError):
        mod1._check_iterfield()
    mod1 = pe.MapNode(EngineTestInterface(), iterfield=['input1', 'input2'], name='mod1')
    mod1.inputs.input1 = [1, 2]
    mod1.inputs.input2 = 3
    with pytest.raises(ValueError):
        mod1._check_iterfield()