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
def test_concurrent_write_as():
    with _get_temp_as_path() as path:
        b = bf.create_context(azure_write_chunk_size=2 ** 20)
        outer_contents = b'miso' * (2 ** 20 + 1)
        inner_contents = b'momo' * (2 ** 20 + 1)
        with pytest.raises(bf.ConcurrentWriteFailure):
            with b.BlobFile(path, 'wb', streaming=True) as f:
                f.write(outer_contents)
                with b.BlobFile(path, 'wb', streaming=True) as f:
                    f.write(inner_contents)
        with b.BlobFile(path, 'rb') as f:
            assert f.read() == inner_contents