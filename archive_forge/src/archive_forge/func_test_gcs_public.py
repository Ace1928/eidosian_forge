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
def test_gcs_public():
    filepath = 'gs://tfds-data/datasets/mnist/3.0.1/dataset_info.json'
    assert bf.exists(filepath)
    assert len(bf.BlobFile(filepath, 'rb').read()) > 0