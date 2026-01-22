import os
import copy
import simplejson
import glob
import os.path as op
from subprocess import Popen
import hashlib
from collections import namedtuple
import pytest
import nipype
import nipype.interfaces.io as nio
from nipype.interfaces.base.traits_extension import isdefined
from nipype.interfaces.base import Undefined, TraitError
from nipype.utils.filemanip import dist_is_editable
from subprocess import check_call, CalledProcessError
@pytest.mark.parametrize('SF_args, inputs_att, expected', [({'templates': templates1}, {'package': 'fsl'}, {'infields': ['package'], 'outfields': ['model', 'preprocess'], 'run_output': {'model': op.join(op.dirname(nipype.__file__), 'interfaces/fsl/model.py'), 'preprocess': op.join(op.dirname(nipype.__file__), 'interfaces/fsl/preprocess.py')}, 'node_output': ['model', 'preprocess']}), ({'templates': templates1, 'force_lists': True}, {'package': 'spm'}, {'infields': ['package'], 'outfields': ['model', 'preprocess'], 'run_output': {'model': [op.join(op.dirname(nipype.__file__), 'interfaces/spm/model.py')], 'preprocess': [op.join(op.dirname(nipype.__file__), 'interfaces/spm/preprocess.py')]}, 'node_output': ['model', 'preprocess']}), ({'templates': templates1}, {'package': 'fsl', 'force_lists': ['model']}, {'infields': ['package'], 'outfields': ['model', 'preprocess'], 'run_output': {'model': [op.join(op.dirname(nipype.__file__), 'interfaces/fsl/model.py')], 'preprocess': op.join(op.dirname(nipype.__file__), 'interfaces/fsl/preprocess.py')}, 'node_output': ['model', 'preprocess']}), ({'templates': templates2}, {'to': 2}, {'infields': ['to'], 'outfields': ['converter'], 'run_output': {'converter': op.join(op.dirname(nipype.__file__), 'interfaces/dcm2nii.py')}, 'node_output': ['converter']}), ({'templates': templates3}, {'package': namedtuple('package', ['name'])('fsl')}, {'infields': ['package'], 'outfields': ['model'], 'run_output': {'model': op.join(op.dirname(nipype.__file__), 'interfaces/fsl/model.py')}, 'node_output': ['model']})])
def test_selectfiles(tmpdir, SF_args, inputs_att, expected):
    tmpdir.chdir()
    base_dir = op.dirname(nipype.__file__)
    dg = nio.SelectFiles(base_directory=base_dir, **SF_args)
    for key, val in inputs_att.items():
        setattr(dg.inputs, key, val)
    assert dg._infields == expected['infields']
    assert sorted(dg._outfields) == expected['outfields']
    assert sorted(dg._outputs().get()) == expected['node_output']
    res = dg.run()
    for key, val in expected['run_output'].items():
        assert getattr(res.outputs, key) == val