import os
from copy import deepcopy
import pytest
from ... import engine as pe
from ....interfaces import base as nib
from ....interfaces import utility as niu
from .... import config
from ..utils import (
def test_clean_working_directory(tmpdir):

    class OutputSpec(nib.TraitedSpec):
        files = nib.traits.List(nib.File)
        others = nib.File()

    class InputSpec(nib.TraitedSpec):
        infile = nib.File()
    outputs = OutputSpec()
    inputs = InputSpec()
    filenames = ['file.hdr', 'file.img', 'file.BRIK', 'file.HEAD', '_0x1234.json', 'foo.txt']
    outfiles = []
    for filename in filenames:
        outfile = tmpdir.join(filename)
        outfile.write('dummy')
        outfiles.append(outfile.strpath)
    outputs.files = outfiles[:4:2]
    outputs.others = outfiles[5]
    inputs.infile = outfiles[-1]
    needed_outputs = ['files']
    config.set_default_config()
    assert os.path.exists(outfiles[5])
    config.set_default_config()
    config.set('execution', 'remove_unnecessary_outputs', False)
    out = clean_working_directory(outputs, tmpdir.strpath, inputs, needed_outputs, deepcopy(config._sections))
    assert os.path.exists(outfiles[5])
    assert out.others == outfiles[5]
    config.set('execution', 'remove_unnecessary_outputs', True)
    out = clean_working_directory(outputs, tmpdir.strpath, inputs, needed_outputs, deepcopy(config._sections))
    assert os.path.exists(outfiles[1])
    assert os.path.exists(outfiles[3])
    assert os.path.exists(outfiles[4])
    assert not os.path.exists(outfiles[5])
    assert out.others == nib.Undefined
    assert len(out.files) == 2
    config.set_default_config()