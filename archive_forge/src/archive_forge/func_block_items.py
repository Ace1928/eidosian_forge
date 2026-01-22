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
def block_items(objekt, block, eldict):
    if objekt not in block:
        if isinstance(objekt.type, PyType):
            if objekt.type not in block:
                block.append(objekt.type)
        block.append(objekt)
        if isinstance(objekt, PyType):
            others = [p for p in eldict.values() if isinstance(p, PyElement) and p.type[1] == objekt.name]
            for item in others:
                if item not in block:
                    block.append(item)
    return block