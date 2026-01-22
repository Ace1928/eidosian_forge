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
def test_outputs_removal(tmpdir):

    def test_function(arg1):
        import os
        file1 = os.path.join(os.getcwd(), 'file1.txt')
        file2 = os.path.join(os.getcwd(), 'file2.txt')
        with open(file1, 'wt') as fp:
            fp.write('%d' % arg1)
        with open(file2, 'wt') as fp:
            fp.write('%d' % arg1)
        return (file1, file2)
    n1 = pe.Node(niu.Function(input_names=['arg1'], output_names=['file1', 'file2'], function=test_function), base_dir=tmpdir.strpath, name='testoutputs')
    n1.inputs.arg1 = 1
    n1.config = {'execution': {'remove_unnecessary_outputs': True}}
    n1.config = merge_dict(deepcopy(config._sections), n1.config)
    n1.run()
    assert tmpdir.join(n1.name, 'file1.txt').check()
    assert tmpdir.join(n1.name, 'file1.txt').check()
    n1.needed_outputs = ['file2']
    n1.run()
    assert not tmpdir.join(n1.name, 'file1.txt').check()
    assert tmpdir.join(n1.name, 'file2.txt').check()