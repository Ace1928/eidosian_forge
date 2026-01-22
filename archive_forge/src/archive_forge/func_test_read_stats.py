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
@pytest.mark.parametrize('buffer_size', [1, 100])
@pytest.mark.parametrize('ctx', [_get_temp_gcs_path, _get_temp_as_path])
def test_read_stats(buffer_size, ctx):
    with ctx() as path:
        contents = b'meow!'
        with bf.BlobFile(path, 'wb') as w:
            w.write(contents)
        with bf.BlobFile(path, 'rb', buffer_size=buffer_size) as r:
            r.read(1)
        if buffer_size == 1:
            assert r.raw.bytes_read == 1
        else:
            assert r.raw.bytes_read == len(contents)
        with bf.BlobFile(path, 'rb', buffer_size=buffer_size) as r:
            r.read(1)
            r.seek(4)
            r.read(1)
            r.seek(1000000)
            assert r.read(1) == b''
        if buffer_size == 1:
            assert r.raw.requests == 2
            assert r.raw.bytes_read == 2
        else:
            assert r.raw.requests == 1
            assert r.raw.bytes_read == len(contents)