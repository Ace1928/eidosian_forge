import os
import nipype.interfaces.fsl.dti as fsl
from nipype.interfaces.fsl import Info, no_fsl
from nipype.interfaces.base import Undefined
import pytest
from nipype.testing.fixtures import create_files_in_directory
@pytest.mark.xfail(reason='These tests are skipped until we clean up some of this code')
def test_Vec_reg():
    vrg = fsl.VecReg()
    assert vrg.cmd == 'vecreg'
    with pytest.raises(ValueError):
        vrg.run()
    vrg.inputs.infile = 'infile'
    vrg.inputs.outfile = 'outfile'
    vrg.inputs.refVolName = 'MNI152'
    vrg.inputs.affineTmat = 'tmat.mat'
    assert vrg.cmdline == 'vecreg -i infile -o outfile -r MNI152 -t tmat.mat'
    vrg2 = fsl.VecReg(infile='infile2', outfile='outfile2', refVolName='MNI152', affineTmat='tmat2.mat', brainMask='nodif_brain_mask')
    actualCmdline = sorted(vrg2.cmdline.split())
    cmd = 'vecreg -i infile2 -o outfile2 -r MNI152 -t tmat2.mat -m nodif_brain_mask'
    desiredCmdline = sorted(cmd.split())
    assert actualCmdline == desiredCmdline
    vrg3 = fsl.VecReg()
    results = vrg3.run(infile='infile3', outfile='outfile3', refVolName='MNI152', affineTmat='tmat3.mat')
    assert results.runtime.cmdline == 'vecreg -i infile3 -o outfile3 -r MNI152 -t tmat3.mat'
    assert results.runtime.returncode != 0
    assert results.interface.inputs.infile == 'infile3'
    assert results.interface.inputs.outfile == 'outfile3'
    assert results.interface.inputs.refVolName == 'MNI152'
    assert results.interface.inputs.affineTmat == 'tmat3.mat'
    opt_map = {'verbose': ('-v', True), 'helpDoc': ('-h', True), 'tensor': ('--tensor', True), 'affineTmat': ('-t Tmat', 'Tmat'), 'warpFile': ('-w wrpFile', 'wrpFile'), 'interpolation': ('--interp=sinc', 'sinc'), 'brainMask': ('-m mask', 'mask')}
    for name, settings in list(opt_map.items()):
        vrg4 = fsl.VecReg(infile='infile', outfile='outfile', refVolName='MNI152', **{name: settings[1]})
        assert vrg4.cmdline == vrg4.cmd + ' -i infile -o outfile -r MNI152 ' + settings[0]