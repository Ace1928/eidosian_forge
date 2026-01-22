from __future__ import annotations
import argparse
import datetime
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Tuple
from rdflib.graph import Graph  # noqa: E402
from rdflib.namespace import DCTERMS, OWL, RDFS, SKOS  # noqa: E402
from rdflib.util import guess_format  # noqa: E402
from rdflib.namespace import DefinedNamespace, Namespace
def validate_object_id(object_id: str) -> None:
    for c in object_id:
        if not c.isupper():
            raise ValueError('The supplied object_id must be an all-capitals string')