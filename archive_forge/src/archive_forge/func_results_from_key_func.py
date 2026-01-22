import logging
from os import mkdir
from os.path import abspath, exists
from threading import Thread
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Optional, Tuple
from urllib.request import pathname2url
from rdflib.store import NO_STORE, VALID_STORE, Store
from rdflib.term import Identifier, Node, URIRef
def results_from_key_func(i: int, from_string: Callable[[bytes], Node]) -> _ResultsFromKeyFunc:

    def from_key(key: bytes, subject: Optional[Node], predicate: Optional[Node], object: Optional[Node], contexts_value: bytes) -> Tuple[Tuple[Node, Node, Node], Generator[Node, None, None]]:
        """Takes a key and subject, predicate, object; returns tuple for yield"""
        parts = key.split('^'.encode('latin-1'))
        if subject is None:
            s = from_string(parts[(3 - i + 0) % 3 + 1])
        else:
            s = subject
        if predicate is None:
            p = from_string(parts[(3 - i + 1) % 3 + 1])
        else:
            p = predicate
        if object is None:
            o = from_string(parts[(3 - i + 2) % 3 + 1])
        else:
            o = object
        return ((s, p, o), (from_string(c) for c in contexts_value.split('^'.encode('latin-1')) if c))
    return from_key