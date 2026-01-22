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
def get_mod(name, path=None):
    mod_a = sys.modules.get(name)
    if not mod_a or not isinstance(mod_a, types.ModuleType):
        mod_a = importlib.import_module(name, path)
        sys.modules[name] = mod_a
    return mod_a