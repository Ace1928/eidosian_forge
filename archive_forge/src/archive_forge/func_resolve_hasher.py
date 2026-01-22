import hashlib
import os
import pathlib
import re
import shutil
import tarfile
import urllib
import warnings
import zipfile
from urllib.request import urlretrieve
from keras.src.api_export import keras_export
from keras.src.backend import config
from keras.src.utils import io_utils
from keras.src.utils.module_utils import gfile
from keras.src.utils.progbar import Progbar
def resolve_hasher(algorithm, file_hash=None):
    """Returns hash algorithm as hashlib function."""
    if algorithm == 'sha256':
        return hashlib.sha256()
    if algorithm == 'auto' and file_hash is not None and (len(file_hash) == 64):
        return hashlib.sha256()
    return hashlib.md5()