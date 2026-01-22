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
def find_and_replace(base, mods):
    base.xmlns_map = mods.xmlns_map
    recursive_add_xmlns_map(base, mods)
    rm = []
    for part in mods.parts:
        try:
            _name = part.name
        except AttributeError:
            continue
        for _part in base.parts:
            try:
                if _name == _part.name:
                    rm.append(_part)
            except AttributeError:
                continue
    for part in rm:
        base.parts.remove(part)
    base.parts.extend(mods.parts)
    return base