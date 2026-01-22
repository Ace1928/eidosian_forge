import io
import os
from ipywidgets import Image
import hashlib
import pkgutil
import tempfile
from contextlib import contextmanager
def test_format_inference_overridable():
    with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as f:
        name = f.name
        f.close()
        img = Image.from_file(name, format='gif')
    assert img.format == 'gif'