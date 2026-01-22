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
def test_bet(setup_infile):
    tmp_infile, tp_dir = setup_infile
    tmp_infile = os.path.relpath(tmp_infile, start=os.getcwd())
    better = fsl.BET()
    assert better.cmd == 'bet'
    with pytest.raises(ValueError):
        better.run()
    better.inputs.in_file = tmp_infile
    outfile = fsl_name(better, 'foo_brain')
    realcmd = 'bet %s %s' % (tmp_infile, outfile)
    assert better.cmdline == realcmd
    outfile = fsl_name(better, '/newdata/bar')
    better.inputs.out_file = outfile
    realcmd = 'bet %s %s' % (tmp_infile, outfile)
    assert better.cmdline == realcmd

    def func():
        better.run(in_file='foo2.nii', out_file='bar.nii')
    with pytest.raises(TraitError):
        func()
    opt_map = {'outline': ('-o', True), 'mask': ('-m', True), 'skull': ('-s', True), 'no_output': ('-n', True), 'frac': ('-f 0.40', 0.4), 'vertical_gradient': ('-g 0.75', 0.75), 'radius': ('-r 20', 20), 'center': ('-c 54 75 80', [54, 75, 80]), 'threshold': ('-t', True), 'mesh': ('-e', True), 'surfaces': ('-A', True)}
    better = fsl.BET()
    outfile = fsl_name(better, 'foo_brain')
    for name, settings in list(opt_map.items()):
        better = fsl.BET(**{name: settings[1]})
        better.inputs.in_file = tmp_infile
        realcmd = ' '.join([better.cmd, tmp_infile, outfile, settings[0]])
        assert better.cmdline == realcmd