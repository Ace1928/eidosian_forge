import logging
from os import mkdir
from os.path import abspath, exists
from threading import Thread
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Optional, Tuple
from urllib.request import pathname2url
from rdflib.store import NO_STORE, VALID_STORE, Store
from rdflib.term import Identifier, Node, URIRef
def readable_index(i: int) -> str:
    s, p, o = '?' * 3
    if i & 1:
        s = 's'
    if i & 2:
        p = 'p'
    if i & 4:
        o = 'o'
    return '%s,%s,%s' % (s, p, o)