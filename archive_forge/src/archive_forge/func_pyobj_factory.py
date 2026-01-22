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
def pyobj_factory(name, value_type, elms=None):
    pyobj = PyObj(name, pyify(name))
    pyobj.value_type = value_type
    if elms:
        if name not in [c.name for c in elms]:
            elms.append(pyobj)
    return pyobj