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
def check_depends(targets, dependencies):
    """Return true if all targets exist and are newer than all dependencies.

    An OSError will be raised if there are missing dependencies.
    """
    tgts = ensure_list(targets)
    deps = ensure_list(dependencies)
    return all(map(op.exists, tgts)) and min(map(op.getmtime, tgts)) > max(list(map(op.getmtime, deps)) + [0])