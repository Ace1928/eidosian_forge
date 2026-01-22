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
def report_extension_load_failures():
    if not _extension_load_failures:
        return
    if config.GlobalConfig().suppress_warning('missing_extensions'):
        return
    from .trace import warning
    warning('brz: warning: some compiled extensions could not be loaded; see ``brz help missing-extensions``')