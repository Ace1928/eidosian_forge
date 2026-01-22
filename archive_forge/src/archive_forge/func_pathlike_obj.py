from struct import pack, unpack, calcsize, error, Struct
import os
import sys
import time
import array
import tempfile
import logging
import io
from datetime import date
import zipfile
def pathlike_obj(path):
    if is_string(path):
        return path
    elif hasattr(path, '__fspath__'):
        return path.__fspath__()
    else:
        try:
            return str(path)
        except:
            return path