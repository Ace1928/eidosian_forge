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
def pyelement_factory(name, value_type, elms=None):
    obj = PyElement(name, pyify(name))
    obj.value_type = value_type
    if elms:
        if name not in [c.name for c in elms]:
            elms.append(obj)
    return obj