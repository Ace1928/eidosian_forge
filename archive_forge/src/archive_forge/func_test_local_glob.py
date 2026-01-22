import datetime
import warnings
import random
import string
import tempfile
import os
import contextlib
import json
import urllib.request
import hashlib
import time
import subprocess as sp
import multiprocessing as mp
import platform
import pickle
import zipfile
import re
import av
import pytest
from tensorflow.io import gfile
import imageio
import numpy as np
import blobfile as bf
from blobfile import _ops as ops, _azure as azure, _common as common
@pytest.mark.parametrize('ctx', [_get_temp_local_path])
def test_local_glob(ctx):
    contents = b'meow!'
    with ctx() as path:
        dirpath = bf.dirname(path)
        a_path = bf.join(dirpath, 'ab')
        with bf.BlobFile(a_path, 'wb') as w:
            w.write(contents)
        b_path = bf.join(dirpath, 'bb')
        with bf.BlobFile(b_path, 'wb') as w:
            w.write(contents)

        def assert_listing_equal(path, desired):
            desired = sorted([bf.join(dirpath, p) for p in desired])
            actual = sorted(list(bf.glob(path)))
            assert actual == desired, f'{actual} != {desired}'
        with chdir(dirpath):
            assert_listing_equal('ab', ['ab'])
            assert_listing_equal('*b', ['ab', 'bb'])
            assert_listing_equal('a*', ['ab'])
            assert_listing_equal('ab*', ['ab'])
            assert_listing_equal('*', ['ab', 'bb'])
            assert_listing_equal('bb', ['bb'])