import io
import os
from ipywidgets import Image
import hashlib
import pkgutil
import tempfile
from contextlib import contextmanager
def assert_equal_hash(byte_str, digest):
    assert get_hash_hex(byte_str) == digest