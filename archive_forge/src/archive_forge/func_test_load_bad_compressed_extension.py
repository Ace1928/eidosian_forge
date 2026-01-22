import pathlib
import shutil
from os.path import dirname
from os.path import join as pjoin
from tempfile import TemporaryDirectory
import numpy as np
from .. import (
from ..filebasedimages import ImageFileError
from ..loadsave import _signature_matches_extension, load, read_img_data
from ..openers import Opener
from ..optpkg import optional_package
from ..testing import deprecated_to, expires
from ..tmpdirs import InTemporaryDirectory
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal
@pytest.mark.parametrize('extension', ['.gz', '.bz2', '.zst'])
def test_load_bad_compressed_extension(tmp_path, extension):
    if extension == '.zst' and (not have_pyzstd):
        pytest.skip()
    file_path = tmp_path / f'img.nii{extension}'
    file_path.write_bytes(b'bad')
    with pytest.raises(ImageFileError, match='.*is not a .* file'):
        load(file_path)