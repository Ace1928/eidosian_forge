import array
import base64
import contextlib
import gc
import io
import math
import os
import shutil
import sys
import tempfile
import cairocffi
import pikepdf
import pytest
from . import (
def test_ps_surface():
    assert set(PSSurface.get_levels()) >= set([cairocffi.PS_LEVEL_2, cairocffi.PS_LEVEL_3])
    assert PSSurface.ps_level_to_string(cairocffi.PS_LEVEL_3) == 'PS Level 3'
    with pytest.raises(TypeError):
        PSSurface.ps_level_to_string('PS_LEVEL_42')
    with pytest.raises(ValueError):
        PSSurface.ps_level_to_string(42)
    with temp_directory() as tempdir:
        filename = os.path.join(tempdir, 'foo.ps')
        filename_bytes = filename.encode(sys.getfilesystemencoding())
        file_obj = io.BytesIO()
        for target in [filename, filename_bytes, file_obj, None]:
            PSSurface(target, 123, 432).finish()
        with open(filename, 'rb') as fd:
            assert fd.read().startswith(b'%!PS')
        with open(filename_bytes, 'rb') as fd:
            assert fd.read().startswith(b'%!PS')
        assert file_obj.getvalue().startswith(b'%!PS')
    file_obj = io.BytesIO()
    surface = PSSurface(file_obj, 1, 1)
    surface.restrict_to_level(cairocffi.PS_LEVEL_2)
    assert surface.get_eps() is False
    surface.set_eps('lol')
    assert surface.get_eps() is True
    surface.set_eps('')
    assert surface.get_eps() is False
    surface.set_size(10, 12)
    surface.dsc_comment('%%Lorem')
    surface.dsc_begin_setup()
    surface.dsc_comment('%%ipsum')
    surface.dsc_begin_page_setup()
    surface.dsc_comment('%%dolor')
    surface.finish()
    ps_bytes = file_obj.getvalue()
    assert b'%%Lorem' in ps_bytes
    assert b'%%ipsum' in ps_bytes
    assert b'%%dolor' in ps_bytes