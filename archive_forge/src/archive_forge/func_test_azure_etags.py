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
@pytest.mark.parametrize('ctx', [_get_temp_as_path])
def test_azure_etags(ctx):
    contents = b'bark!'
    alternative_contents = b'ruff'
    with ctx() as path:
        bf.BlobFile(path, 'wb').write(contents)
        version = bf.stat(path).version
        with bf.BlobFile(path, 'wb', version=version) as f:
            for _ in range(1000):
                f.write(alternative_contents)
        version = bf.stat(path).version
        with bf.BlobFile(path, 'wb', version=version) as f:
            f.write(alternative_contents)
        new_version = bf.stat(path).version
        assert new_version != version
        with pytest.raises(bf.VersionMismatch):
            with bf.BlobFile(path, 'wb', version=version) as f:
                f.write(contents)
        assert bf.BlobFile(path, 'rb').read() == alternative_contents
        bf.remove(path)