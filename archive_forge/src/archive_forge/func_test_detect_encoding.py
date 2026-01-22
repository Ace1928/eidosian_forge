import io
import os.path
from IPython.utils import openpy
def test_detect_encoding():
    with open(nonascii_path, 'rb') as f:
        enc, lines = openpy.detect_encoding(f.readline)
    assert enc == 'iso-8859-5'