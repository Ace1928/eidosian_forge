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
def test_copy_azure_public():
    with _get_temp_as_path() as dst:
        bf.copy(AZURE_PUBLIC_URL, dst)
        assert _read_contents(dst)[:4] == AZURE_PUBLIC_URL_HEADER