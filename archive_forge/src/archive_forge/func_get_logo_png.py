import io
import os
from ipywidgets import Image
import hashlib
import pkgutil
import tempfile
from contextlib import contextmanager
@contextmanager
def get_logo_png():
    LOGO_DATA = pkgutil.get_data('ipywidgets.widgets.tests', 'data/jupyter-logo-transparent.png')
    handle, fname = tempfile.mkstemp()
    os.close(handle)
    with open(fname, 'wb') as f:
        f.write(LOGO_DATA)
    yield fname
    os.remove(fname)