from __future__ import annotations
import datetime as py_datetime  # naming conflict with function within this module
import hashlib
import math
import operator as pyop  # python operators
import random
import re
import uuid
import warnings
from decimal import ROUND_HALF_DOWN, ROUND_HALF_UP, Decimal, InvalidOperation
from functools import reduce
from typing import Any, Callable, Dict, NoReturn, Optional, Tuple, Union, overload
from urllib.parse import quote
import isodate
from pyparsing import ParseResults
from rdflib.namespace import RDF, XSD
from rdflib.plugins.sparql.datatypes import (
from rdflib.plugins.sparql.parserutils import CompValue, Expr
from rdflib.plugins.sparql.sparql import (
from rdflib.term import (
def unregister_custom_function(uri: URIRef, func: Optional[Callable[..., Any]]=None) -> None:
    """
    The 'func' argument is included for compatibility with existing code.
    A previous implementation checked that the function associated with
    the given uri was actually 'func', but this is not necessary as the
    uri should uniquely identify the function.
    """
    if _CUSTOM_FUNCTIONS.get(uri):
        del _CUSTOM_FUNCTIONS[uri]
    else:
        warnings.warn('This function is not registered as %s' % uri.n3())