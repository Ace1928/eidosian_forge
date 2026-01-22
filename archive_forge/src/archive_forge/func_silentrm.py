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
def silentrm(filename):
    """
    Equivalent to ``rm -f``, returns ``False`` if the file did not
    exist.

    Parameters
    ----------

    filename : str
        file to be deleted

    """
    try:
        os.remove(filename)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise
        return False
    return True