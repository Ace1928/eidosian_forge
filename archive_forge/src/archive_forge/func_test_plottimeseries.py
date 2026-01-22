import os
import numpy as np
import nibabel as nb
import pytest
import nipype.interfaces.fsl.utils as fsl
from nipype.interfaces.fsl import no_fsl, Info
from nipype.testing.fixtures import create_files_in_directory_plus_output_type
@pytest.mark.skipif(no_fsl(), reason='fsl is not installed')
def test_plottimeseries(create_files_in_directory_plus_output_type):
    filelist, outdir, _ = create_files_in_directory_plus_output_type
    parfiles = create_parfiles()
    plotter = fsl.PlotTimeSeries()
    assert plotter.cmd == 'fsl_tsplot'
    with pytest.raises(ValueError):
        plotter.run()
    plotter.inputs.in_file = parfiles[0]
    plotter.inputs.labels = ['x', 'y', 'z']
    plotter.inputs.y_range = (0, 1)
    plotter.inputs.title = 'test plot'
    plotter.inputs.out_file = 'foo.png'
    assert plotter.cmdline == "fsl_tsplot -i %s -a x,y,z -o foo.png -t 'test plot' -u 1 --ymin=0 --ymax=1" % parfiles[0]
    plotter2 = fsl.PlotTimeSeries(in_file=parfiles, title='test2 plot', plot_range=(2, 5), out_file='bar.png')
    assert plotter2.cmdline == "fsl_tsplot -i %s,%s -o bar.png --start=2 --finish=5 -t 'test2 plot' -u 1" % tuple(parfiles)