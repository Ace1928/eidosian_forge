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
def test_read_with_size(ctx):
    contents = b'meow!\npurr\n'
    with ctx() as path:
        path = bf.join(path, 'a folder', 'a.file')
        bf.makedirs(bf.dirname(path))
        with bf.BlobFile(path, 'wb') as w:
            w.write(contents)
        with bf.BlobFile(path, 'rb', file_size=1) as r:
            assert r.read() == contents[:1]
            assert r.tell() == 1