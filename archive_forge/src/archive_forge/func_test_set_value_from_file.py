import io
import os
from ipywidgets import Image
import hashlib
import pkgutil
import tempfile
from contextlib import contextmanager
def test_set_value_from_file():
    img = Image()
    with get_logo_png() as LOGO_PNG:
        with open(LOGO_PNG, 'rb') as f:
            img.set_value_from_file(f)
            assert_equal_hash(img.value, LOGO_PNG_DIGEST)