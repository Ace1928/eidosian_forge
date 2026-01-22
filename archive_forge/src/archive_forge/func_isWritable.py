import reportlab
import sys, os, fnmatch, re, functools
from configparser import ConfigParser
import unittest
from reportlab.lib.utils import isCompactDistro, __rl_loader__, rl_isdir, asUnicode
def isWritable(D):
    try:
        fn = '00DELETE.ME'
        f = open(fn, 'w')
        f.write('test of writability - can be deleted')
        f.close()
        if os.path.isfile(fn):
            os.remove(fn)
            return 1
    except:
        return 0