import reportlab
import sys, os, fnmatch, re, functools
from configparser import ConfigParser
import unittest
from reportlab.lib.utils import isCompactDistro, __rl_loader__, rl_isdir, asUnicode
def mockUrlRead(name):
    if name in _mockumap:
        with open(os.path.join(testsFolder, os.path.basename(name)), 'rb') as f:
            return f.read()
    else:
        from urllib.request import urlopen
        return urlopen(name).read()