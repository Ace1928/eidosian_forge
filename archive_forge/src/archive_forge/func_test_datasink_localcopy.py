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
def test_datasink_localcopy(dummy_input, tmpdir):
    """
    Function to validate DataSink will make local copy via local_copy
    attribute
    """
    local_dir = tmpdir.strpath
    container = 'outputs'
    attr_folder = 'text_file'
    input_path = dummy_input
    ds = nio.DataSink()
    ds.inputs.container = container
    ds.inputs.local_copy = local_dir
    setattr(ds.inputs, attr_folder, input_path)
    local_copy = os.path.join(local_dir, container, attr_folder, os.path.basename(input_path))
    ds.run()
    src_md5 = hashlib.md5(open(input_path, 'rb').read()).hexdigest()
    dst_md5 = hashlib.md5(open(local_copy, 'rb').read()).hexdigest()
    assert src_md5 == dst_md5