import io
import os
from ipywidgets import Image
import hashlib
import pkgutil
import tempfile
from contextlib import contextmanager
def test_from_url_unicode():
    img = Image.from_url('https://jupyter.org/assets/main-logo.svg')
    assert img.value == b'https://jupyter.org/assets/main-logo.svg'