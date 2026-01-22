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
def recursive_add_xmlns_map(_sch, base):
    for _part in _sch.parts:
        _part.xmlns_map.update(base.xmlns_map)
        if isinstance(_part, Complex):
            recursive_add_xmlns_map(_part, base)