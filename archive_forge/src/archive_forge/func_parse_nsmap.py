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
def parse_nsmap(fil):
    events = ('start', 'start-ns', 'end-ns')
    root = None
    ns_map = []
    for event, elem in ElementTree.iterparse(fil, events):
        if event == 'start-ns':
            ns_map.append(elem)
        elif event == 'end-ns':
            ns_map.pop()
        elif event == 'start':
            if root is None:
                root = elem
            elem.set(NS_MAP, dict(ns_map))
    return ElementTree.ElementTree(root)