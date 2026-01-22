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
def toProperty(label):
    """
    CamelCase + lowercase initial a string


    FIRST_NM => firstNm

    firstNm => firstNm

    """
    label = re.sub('[^\\w]', ' ', label)
    label = re.sub('([a-z])([A-Z])', '\\1 \\2', label)
    label = label.split(' ')
    return ''.join([label[0].lower()] + [x.capitalize() for x in label[1:]])