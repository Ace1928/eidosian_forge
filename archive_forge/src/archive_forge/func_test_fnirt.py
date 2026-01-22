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
def test_fnirt(setup_flirt):
    tmpdir, infile, reffile = setup_flirt
    tmpdir.chdir()
    fnirt = fsl.FNIRT()
    assert fnirt.cmd == 'fnirt'
    params = [('subsampling_scheme', '--subsamp', [4, 2, 2, 1], '4,2,2,1'), ('max_nonlin_iter', '--miter', [4, 4, 4, 2], '4,4,4,2'), ('ref_fwhm', '--reffwhm', [4, 2, 2, 0], '4,2,2,0'), ('in_fwhm', '--infwhm', [4, 2, 2, 0], '4,2,2,0'), ('apply_refmask', '--applyrefmask', [0, 0, 1, 1], '0,0,1,1'), ('apply_inmask', '--applyinmask', [0, 0, 0, 1], '0,0,0,1'), ('regularization_lambda', '--lambda', [0.5, 0.75], '0.5,0.75'), ('intensity_mapping_model', '--intmod', 'global_non_linear', 'global_non_linear')]
    for item, flag, val, strval in params:
        fnirt = fsl.FNIRT(in_file=infile, ref_file=reffile, **{item: val})
        log = fnirt._gen_fname(infile, suffix='_log.txt', change_ext=False)
        iout = fnirt._gen_fname(infile, suffix='_warped')
        if item in 'max_nonlin_iter':
            cmd = 'fnirt --in=%s --logout=%s %s=%s --ref=%s --iout=%s' % (infile, log, flag, strval, reffile, iout)
        elif item in ('in_fwhm', 'intensity_mapping_model'):
            cmd = 'fnirt --in=%s %s=%s --logout=%s --ref=%s --iout=%s' % (infile, flag, strval, log, reffile, iout)
        elif item.startswith('apply'):
            cmd = 'fnirt %s=%s --in=%s --logout=%s --ref=%s --iout=%s' % (flag, strval, infile, log, reffile, iout)
        else:
            cmd = 'fnirt --in=%s --logout=%s --ref=%s %s=%s --iout=%s' % (infile, log, reffile, flag, strval, iout)
        assert fnirt.cmdline == cmd
    fnirt = fsl.FNIRT()
    with pytest.raises(ValueError):
        fnirt.run()
    fnirt.inputs.in_file = infile
    fnirt.inputs.ref_file = reffile
    intmap_basename = '%s_intmap' % fsl.FNIRT.intensitymap_file_basename(infile)
    intmap_image = fsl_name(fnirt, intmap_basename)
    intmap_txt = '%s.txt' % intmap_basename
    with open(intmap_image, 'w'):
        pass
    with open(intmap_txt, 'w'):
        pass
    opt_map = [('affine_file', '--aff=%s' % infile, infile), ('inwarp_file', '--inwarp=%s' % infile, infile), ('in_intensitymap_file', '--intin=%s' % intmap_basename, [intmap_image]), ('in_intensitymap_file', '--intin=%s' % intmap_basename, [intmap_image, intmap_txt]), ('config_file', '--config=%s' % infile, infile), ('refmask_file', '--refmask=%s' % infile, infile), ('inmask_file', '--inmask=%s' % infile, infile), ('field_file', '--fout=%s' % infile, infile), ('jacobian_file', '--jout=%s' % infile, infile), ('modulatedref_file', '--refout=%s' % infile, infile), ('out_intensitymap_file', '--intout=%s' % intmap_basename, True), ('out_intensitymap_file', '--intout=%s' % intmap_basename, intmap_image), ('fieldcoeff_file', '--cout=%s' % infile, infile), ('log_file', '--logout=%s' % infile, infile)]
    for name, settings, arg in opt_map:
        fnirt = fsl.FNIRT(in_file=infile, ref_file=reffile, **{name: arg})
        if name in ('config_file', 'affine_file', 'field_file', 'fieldcoeff_file'):
            cmd = 'fnirt %s --in=%s --logout=%s --ref=%s --iout=%s' % (settings, infile, log, reffile, iout)
        elif name in 'refmask_file':
            cmd = 'fnirt --in=%s --logout=%s --ref=%s %s --iout=%s' % (infile, log, reffile, settings, iout)
        elif name in ('in_intensitymap_file', 'inwarp_file', 'inmask_file', 'jacobian_file'):
            cmd = 'fnirt --in=%s %s --logout=%s --ref=%s --iout=%s' % (infile, settings, log, reffile, iout)
        elif name in 'log_file':
            cmd = 'fnirt --in=%s %s --ref=%s --iout=%s' % (infile, settings, reffile, iout)
        else:
            cmd = 'fnirt --in=%s --logout=%s %s --ref=%s --iout=%s' % (infile, log, settings, reffile, iout)
        assert fnirt.cmdline == cmd
        if name == 'out_intensitymap_file':
            assert fnirt._list_outputs()['out_intensitymap_file'] == [intmap_image, intmap_txt]