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
def getArgvDict(**kw):
    """ Builds a dictionary from its keyword arguments with overrides from sys.argv.
        Attempts to be smart about conversions, but the value can be an instance
        of ArgDictValue to allow specifying a conversion function.
    """

    def handleValue(v, av, func):
        if func:
            v = func(av)
        elif isStr(v):
            v = av
        elif isinstance(v, float):
            v = float(av)
        elif isinstance(v, int):
            v = int(av)
        elif isinstance(v, list):
            v = list(literal_eval(av), {})
        elif isinstance(v, tuple):
            v = tuple(literal_eval(av), {})
        else:
            raise TypeError("Can't convert string %r to %s" % (av, type(v)))
        return v
    A = sys.argv[1:]
    R = {}
    for k, v in kw.items():
        if isinstance(v, ArgvDictValue):
            v, func = (v.value, v.func)
        else:
            func = None
        handled = 0
        ke = k + '='
        for a in A:
            if a.startswith(ke):
                av = a[len(ke):]
                A.remove(a)
                R[k] = handleValue(v, av, func)
                handled = 1
                break
        if not handled:
            R[k] = handleValue(v, v, func)
    return R