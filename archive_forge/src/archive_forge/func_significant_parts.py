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
def significant_parts(self):
    res = []
    for p in self.parts:
        if isinstance(p, Annotation):
            continue
        else:
            res.append(p)
    return res