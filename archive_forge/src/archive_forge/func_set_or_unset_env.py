import codecs
import errno
import os
import re
import stat
import sys
import time
from functools import partial
from typing import Dict, List
from .lazy_import import lazy_import
import locale
import ntpath
import posixpath
import select
import shutil
from shutil import rmtree
import socket
import subprocess
import unicodedata
from breezy import (
from breezy.i18n import gettext
from hashlib import md5
from hashlib import sha1 as sha
import breezy
from . import errors
def set_or_unset_env(env_variable, value):
    """Modify the environment, setting or removing the env_variable.

    :param env_variable: The environment variable in question
    :param value: The value to set the environment to. If None, then
        the variable will be removed.
    :return: The original value of the environment variable.
    """
    orig_val = os.environ.get(env_variable)
    if value is None:
        if orig_val is not None:
            del os.environ[env_variable]
    else:
        os.environ[env_variable] = value
    return orig_val