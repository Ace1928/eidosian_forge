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
@classmethod
def check_pil_image_size(cls, im):
    max_image_size = cls._max_image_size
    if max_image_size is None:
        return
    w, h = im.size
    m = im.mode
    size = max(1, (1 if m == '1' else 8 * len(m)) * w * h >> 3)
    if size > max_image_size:
        raise MemoryError('PIL %s %s x %s image would use %s > %s bytes' % (m, w, h, size, max_image_size))