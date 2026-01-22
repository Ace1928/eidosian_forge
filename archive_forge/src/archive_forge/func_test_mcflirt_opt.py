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
def test_mcflirt_opt(setup_flirt):
    tmpdir, infile, reffile = setup_flirt
    _, nme = os.path.split(infile)
    opt_map = {'cost': ('-cost mutualinfo', 'mutualinfo'), 'bins': ('-bins 256', 256), 'dof': ('-dof 6', 6), 'ref_vol': ('-refvol 2', 2), 'scaling': ('-scaling 6.00', 6.0), 'smooth': ('-smooth 1.00', 1.0), 'rotation': ('-rotation 2', 2), 'stages': ('-stages 3', 3), 'init': ('-init %s' % infile, infile), 'use_gradient': ('-gdt', True), 'use_contour': ('-edge', True), 'mean_vol': ('-meanvol', True), 'stats_imgs': ('-stats', True), 'save_mats': ('-mats', True), 'save_plots': ('-plots', True)}
    for name, settings in list(opt_map.items()):
        fnt = fsl.MCFLIRT(in_file=infile, **{name: settings[1]})
        outfile = os.path.join(os.getcwd(), nme)
        outfile = fnt._gen_fname(outfile, suffix='_mcf')
        instr = '-in %s' % infile
        outstr = '-out %s' % outfile
        if name in ('init', 'cost', 'dof', 'mean_vol', 'bins'):
            assert fnt.cmdline == ' '.join([fnt.cmd, instr, settings[0], outstr])
        else:
            assert fnt.cmdline == ' '.join([fnt.cmd, instr, outstr, settings[0]])