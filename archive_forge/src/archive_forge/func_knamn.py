import errno
import getopt
import importlib
import re
import sys
import time
import types
from xml.etree import ElementTree as ElementTree
import saml2
from saml2 import SamlBase
def knamn(self, sup, cdict):
    cname = cdict[sup].class_name
    if not cname:
        namesp, tag = cdict[sup].name.split('.')
        if namesp:
            ctag = self.root.modul[namesp].factory(tag).__class__.__name__
            cname = f'{namesp}.{ctag}'
        else:
            cname = tag + '_'
    return cname