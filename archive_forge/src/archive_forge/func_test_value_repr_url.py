import io
import os
from ipywidgets import Image
import hashlib
import pkgutil
import tempfile
from contextlib import contextmanager
def test_value_repr_url():
    img = Image.from_url(b'https://jupyter.org/assets/main-logo.svg')
    assert 'https://jupyter.org/assets/main-logo.svg' in img.__repr__()