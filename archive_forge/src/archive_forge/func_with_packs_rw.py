from gitdb import OStream
import sys
import random
from array import array
from io import BytesIO
import glob
import unittest
import tempfile
import shutil
import os
import gc
import logging
from functools import wraps
def with_packs_rw(func):
    """Function that provides a path into which the packs for testing should be
    copied. Will pass on the path to the actual function afterwards"""

    def wrapper(self, path):
        src_pack_glob = fixture_path('packs/*')
        copy_files_globbed(src_pack_glob, path, hard_link_ok=True)
        return func(self, path)
    wrapper.__name__ = func.__name__
    return wrapper