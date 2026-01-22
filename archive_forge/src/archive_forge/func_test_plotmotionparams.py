import os
import numpy as np
import nibabel as nb
import pytest
import nipype.interfaces.fsl.utils as fsl
from nipype.interfaces.fsl import no_fsl, Info
from nipype.testing.fixtures import create_files_in_directory_plus_output_type
@pytest.mark.skipif(no_fsl(), reason='fsl is not installed')
def test_plotmotionparams(create_files_in_directory_plus_output_type):
    filelist, outdir, _ = create_files_in_directory_plus_output_type
    parfiles = create_parfiles()
    plotter = fsl.PlotMotionParams()
    assert plotter.cmd == 'fsl_tsplot'
    with pytest.raises(ValueError):
        plotter.run()
    plotter.inputs.in_file = parfiles[0]
    plotter.inputs.in_source = 'fsl'
    plotter.inputs.plot_type = 'rotations'
    plotter.inputs.out_file = 'foo.png'
    assert plotter.cmdline == "fsl_tsplot -i %s -o foo.png -t 'MCFLIRT estimated rotations (radians)' --start=1 --finish=3 -a x,y,z" % parfiles[0]
    plotter2 = fsl.PlotMotionParams(in_file=parfiles[1], in_source='spm', plot_type='translations', out_file='bar.png')
    assert plotter2.cmdline == "fsl_tsplot -i %s -o bar.png -t 'Realign estimated translations (mm)' --start=1 --finish=3 -a x,y,z" % parfiles[1]