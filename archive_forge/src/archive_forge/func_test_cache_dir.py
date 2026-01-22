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
@pytest.mark.parametrize('ctx', [_get_temp_gcs_path, _get_temp_as_path])
def test_cache_dir(ctx):
    cache_dir = tempfile.mkdtemp()
    contents = b'meow!'
    alternative_contents = b'purr!'
    with ctx() as path:
        with bf.BlobFile(path, mode='wb') as f:
            f.write(contents)
        with bf.BlobFile(path, mode='rb', streaming=False, cache_dir=cache_dir) as f:
            assert f.read() == contents
        content_hash = hashlib.md5(contents).hexdigest()
        cache_path = bf.join(cache_dir, content_hash, bf.basename(path))
        with open(cache_path, 'rb') as f:
            assert f.read() == contents
        with open(cache_path, 'wb') as f:
            f.write(alternative_contents)
        with bf.BlobFile(path, mode='rb', streaming=False, cache_dir=cache_dir) as f:
            assert f.read() == alternative_contents
        with bf.BlobFile(f'https://{AS_EXTERNAL_ACCOUNT}.blob.core.windows.net/publiccontainer/test_cat_no_md5.png', mode='rb', streaming=False, cache_dir=cache_dir) as f:
            assert len(f.read()) > 0