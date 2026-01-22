from __future__ import annotations
import codecs
import configparser
import csv
import datetime
import fileinput
import getopt
import re
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote
import rdflib
from rdflib.namespace import RDF, RDFS, split_uri
from rdflib.term import URIRef
from the headers
def prefixuri(x, prefix, class_: Optional[URIRef]=None):
    if prefix:
        r = rdflib.URIRef(prefix + quote(x.encode('utf8').replace(' ', '_'), safe=''))
    else:
        r = rdflib.URIRef(x)
    uris[x] = (r, class_)
    return r