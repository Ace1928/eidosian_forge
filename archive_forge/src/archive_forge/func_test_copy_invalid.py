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
@pytest.mark.parametrize('parallel', [False, True])
def test_copy_invalid(parallel):
    for contents in [b'', b'meow!', b'meow!' * (2 * 2 ** 20)]:
        with _get_temp_local_path() as local_path, _get_temp_as_path() as as_path1, _get_temp_as_path() as as_path2:
            invalid_container_as_path = _convert_az_to_https(bf.join(AZURE_INVALID_CONTAINER, 'file.bin'))
            invalid_account_as_path = _convert_az_to_https(bf.join(AZURE_INVALID_CONTAINER_NO_ACCOUNT, 'file.bin'))
            _write_contents(local_path, contents)
            bf.copy(local_path, as_path1, parallel=parallel)
            with pytest.raises(FileNotFoundError, match=invalid_container_as_path):
                bf.copy(local_path, invalid_container_as_path, parallel=parallel)
            with pytest.raises(FileNotFoundError, match=invalid_account_as_path):
                bf.copy(local_path, invalid_account_as_path, parallel=parallel)
            with pytest.raises(FileNotFoundError, match=invalid_container_as_path):
                bf.copy(as_path1, invalid_container_as_path, parallel=parallel)
            with pytest.raises(FileNotFoundError, match=invalid_account_as_path):
                bf.copy(as_path1, invalid_account_as_path, parallel=parallel)
            with pytest.raises(FileNotFoundError, match=invalid_container_as_path):
                bf.copy(invalid_container_as_path, as_path2, parallel=parallel)