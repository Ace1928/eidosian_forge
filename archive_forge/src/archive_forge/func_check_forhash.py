import sys
import pickle
import errno
import subprocess as sp
import gzip
import hashlib
import locale
from hashlib import md5
import os
import os.path as op
import re
import shutil
import contextlib
import posixpath
from pathlib import Path
import simplejson as json
from time import sleep, time
from .. import logging, config, __version__ as version
from .misc import is_container
def check_forhash(filename):
    """checks if file has a hash in its filename"""
    if isinstance(filename, list):
        filename = filename[0]
    path, name = op.split(filename)
    if re.search('(_0x[a-z0-9]{32})', name):
        hashvalue = re.findall('(_0x[a-z0-9]{32})', name)
        return (True, hashvalue)
    else:
        return (False, None)