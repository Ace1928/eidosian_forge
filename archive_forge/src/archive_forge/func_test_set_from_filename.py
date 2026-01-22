import io
import os
from ipywidgets import Image
import hashlib
import pkgutil
import tempfile
from contextlib import contextmanager
def test_set_from_filename():
    img = Image()
    with get_logo_png() as LOGO_PNG:
        img.set_value_from_file(LOGO_PNG)
        assert_equal_hash(img.value, LOGO_PNG_DIGEST)