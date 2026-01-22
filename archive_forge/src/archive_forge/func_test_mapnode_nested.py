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
def test_mapnode_nested(tmpdir):
    tmpdir.chdir()
    from nipype import MapNode, Function

    def func1(in1):
        return in1 + 1
    n1 = MapNode(Function(input_names=['in1'], output_names=['out'], function=func1), iterfield=['in1'], nested=True, name='n1')
    n1.inputs.in1 = [[1, [2]], 3, [4, 5]]
    n1.run()
    assert n1.get_output('out') == [[2, [3]], 4, [5, 6]]
    n2 = MapNode(Function(input_names=['in1'], output_names=['out'], function=func1), iterfield=['in1'], nested=False, name='n1')
    n2.inputs.in1 = [[1, [2]], 3, [4, 5]]
    with pytest.raises(Exception) as excinfo:
        n2.run()
    assert 'can only concatenate list' in str(excinfo.value)