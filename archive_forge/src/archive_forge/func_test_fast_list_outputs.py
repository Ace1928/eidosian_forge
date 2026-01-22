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
def test_fast_list_outputs(setup_infile, tmpdir):
    """By default (no -o), FSL's fast command outputs files into the same
    directory as the input files. If the flag -o is set, it outputs files into
    the cwd"""

    def _run_and_test(opts, output_base):
        outputs = fsl.FAST(**opts)._list_outputs()
        for output in outputs.values():
            if output:
                for filename in ensure_list(output):
                    assert os.path.realpath(filename).startswith(os.path.realpath(output_base))
    tmp_infile, indir = setup_infile
    cwd = tmpdir.mkdir('new')
    cwd.chdir()
    assert indir != cwd.strpath
    out_basename = 'a_basename'
    opts = {'in_files': tmp_infile}
    input_path, input_filename, input_ext = split_filename(tmp_infile)
    _run_and_test(opts, os.path.join(input_path, input_filename))
    opts['out_basename'] = out_basename
    _run_and_test(opts, os.path.join(cwd.strpath, out_basename))