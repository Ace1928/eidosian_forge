import os, pickle, sys, time, types, datetime, importlib
from ast import literal_eval
from base64 import decodebytes as base64_decodebytes, encodebytes as base64_encodebytes
from io import BytesIO
from hashlib import md5
from reportlab.lib.rltempfile import get_rl_tempfile, get_rl_tempdir
from . rl_safe_eval import rl_safe_exec, rl_safe_eval, safer_globals, rl_extended_literal_eval
from PIL import Image
import builtins
import reportlab
import glob, fnmatch
from urllib.parse import unquote, urlparse
from urllib.request import urlopen
from importlib import util as importlib_util
import itertools
def rl_listdir(pn, os_path_isdir=os.path.isdir, os_path_normpath=os.path.normpath, os_listdir=os.listdir):
    if os_path_isdir(pn) or _isFSD or __rl_loader__ is None:
        return os_listdir(pn)
    pn = _startswith_rl(os_path_normpath(pn))
    if not pn.endswith(os.sep):
        pn += os.sep
    return [x[len(pn):] for x in __rl_loader__._files.keys() if x.startswith(pn)]