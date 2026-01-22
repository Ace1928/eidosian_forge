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
def commajoin(l):
    """
    Inverse of commasplit, except that whitespace around items is not conserved.
    Adds more whitespace than needed for simplicity and performance.
    
    >>> commasplit(commajoin(['a', 'b', 'c'])) == [u'a', u'b', u'c']
    True
    >>> commasplit((commajoin([u'a,', u' b ', u'c'])) == [u'a,', u'b', u'c']
    True
    >>> commasplit((commajoin([u'a ', u',b', u'c'])) == [u'a', u',b', u'c'] 
    """
    return u','.join([u' ' + asUnicode(i).replace(u',', u',,') + u' ' for i in l])