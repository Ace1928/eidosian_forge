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
def test_deadline(ctx):
    contents = b'meow'
    with ctx() as path:
        _write_contents(path, contents)
        deadline = Deadline()
        bf_ctx = bf.create_context(get_deadline=deadline.get_deadline)
        deadline.set_deadline(time.time() + 5)
        with bf_ctx.BlobFile(path, 'rb') as f:
            f.read()
        time.sleep(5)
        with pytest.raises(bf.DeadlineExceeded):
            with bf_ctx.BlobFile(path, 'rb') as f:
                f.read()
        deadline.set_deadline(None)
        with bf_ctx.BlobFile(path, 'rb') as f:
            f.read()