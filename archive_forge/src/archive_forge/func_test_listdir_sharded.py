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
@pytest.mark.parametrize('ctx', [_get_temp_local_path, _get_temp_gcs_path, _get_temp_as_path])
def test_listdir_sharded(ctx):
    contents = b'meow!'
    with ctx() as path:
        dirpath = bf.dirname(path)
        with bf.BlobFile(bf.join(dirpath, 'a'), 'wb') as w:
            w.write(contents)
        with bf.BlobFile(bf.join(dirpath, 'aa'), 'wb') as w:
            w.write(contents)
        with bf.BlobFile(bf.join(dirpath, 'b'), 'wb') as w:
            w.write(contents)
        with bf.BlobFile(bf.join(dirpath, 'ca'), 'wb') as w:
            w.write(contents)
        bf.makedirs(bf.join(dirpath, 'c'))
        with bf.BlobFile(bf.join(dirpath, 'c/a'), 'wb') as w:
            w.write(contents)
        assert sorted(list(bf.listdir(dirpath, shard_prefix_length=1))) == ['a', 'aa', 'b', 'c', 'ca']