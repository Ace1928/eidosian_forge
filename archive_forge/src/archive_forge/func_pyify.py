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
def pyify(name):
    res = []
    upc = []
    pre = ''
    for char in name:
        if 'A' <= char <= 'Z':
            upc.append(char)
        elif char == '-':
            upc.append('_')
        else:
            if upc:
                if len(upc) == 1:
                    res.append(pre + upc[0].lower())
                else:
                    if pre:
                        res.append(pre)
                    for uch in upc[:-1]:
                        res.append(uch.lower())
                    res.append('_' + upc[-1].lower())
                upc = []
            res.append(char)
            pre = '_'
    if upc:
        if len(upc) == len(name):
            return name.lower()
        else:
            res.append('_' + ''.join(upc).lower())
    return ''.join(res)