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
def filter_safe_paths(members):
    base_dir = resolve_path('.')
    for finfo in members:
        valid_path = False
        if is_path_in_dir(finfo.name, base_dir):
            valid_path = True
            yield finfo
        elif finfo.issym() or finfo.islnk():
            if is_link_in_dir(finfo, base_dir):
                valid_path = True
                yield finfo
        if not valid_path:
            warnings.warn(f"Skipping invalid path during archive extraction: '{finfo.name}'.", stacklevel=2)