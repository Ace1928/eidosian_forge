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
def pyify_0(name):
    res = ''
    match = re.match('^(([A-Z])[a-z]+)(([A-Z])[a-z]+)?(([A-Z])[a-z]+)?(([A-Z])[a-z]+)?', name)
    res += match.group(1).lower()
    for num in range(3, len(match.groups()), 2):
        try:
            res += '_' + match.group(num + 1).lower() + match.group(num)[1:]
        except AttributeError:
            break
    res = res.replace('-', '_')
    if res in ['class']:
        res += '_'
    return res