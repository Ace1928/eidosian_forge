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
@pytest.fixture(scope='session', autouse=True)
def setup_gcloud_auth():
    if platform.system() == 'Linux' and 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
        sp.run(['gcloud', 'auth', 'activate-service-account', f'--key-file={os.environ['GOOGLE_APPLICATION_CREDENTIALS']}'])
    yield