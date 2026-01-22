from __future__ import annotations
import pathlib
from typing import IO, TYPE_CHECKING, Any, Optional, TextIO, Tuple, Union
from io import TextIOBase, TextIOWrapper
from posixpath import normpath, sep
from urllib.parse import urljoin, urlsplit, urlunsplit
from rdflib.parser import (
def split_iri(iri: str) -> Tuple[str, Optional[str]]:
    for delim in VOCAB_DELIMS:
        at = iri.rfind(delim)
        if at > -1:
            return (iri[:at + 1], iri[at + 1:])
    return (iri, None)