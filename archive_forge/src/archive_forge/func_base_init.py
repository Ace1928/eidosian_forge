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
def base_init(imports):
    line = []
    indent4 = INDENT + INDENT + INDENT + INDENT
    if not imports:
        line.append(f'{INDENT + INDENT}SamlBase.__init__(self, ')
        for attr in BASE_ELEMENT:
            if attr in PROTECTED_KEYWORDS:
                _name = attr + '_'
            else:
                _name = attr
            line.append(f'{indent4}{_name}={_name},')
        line.append(f'{indent4})')
    else:
        for sup, elems in imports.items():
            line.append(f'{INDENT + INDENT}{sup}.__init__(self, ')
            lattr = elems[:]
            lattr.extend(BASE_ELEMENT)
            for attr in lattr:
                if attr in PROTECTED_KEYWORDS:
                    _name = attr + '_'
                else:
                    _name = attr
                line.append(f'{indent4}{_name}={_name},')
            line.append(f'{indent4})')
    return line