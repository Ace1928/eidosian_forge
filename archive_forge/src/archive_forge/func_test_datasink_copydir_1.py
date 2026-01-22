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
def test_datasink_copydir_1(_temp_analyze_files, tmpdir):
    orig_img, orig_hdr = _temp_analyze_files
    outdir = tmpdir
    pth, fname = os.path.split(orig_img)
    ds = nio.DataSink(base_directory=outdir.mkdir('basedir').strpath, parameterization=False)
    setattr(ds.inputs, '@outdir', pth)
    ds.run()
    sep = os.path.sep
    assert tmpdir.join('basedir', pth.split(sep)[-1], fname).check()