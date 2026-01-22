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
@custom_function(XSD.string, raw=True)
@custom_function(XSD.dateTime, raw=True)
@custom_function(XSD.float, raw=True)
@custom_function(XSD.double, raw=True)
@custom_function(XSD.decimal, raw=True)
@custom_function(XSD.integer, raw=True)
@custom_function(XSD.boolean, raw=True)
def default_cast(e: Expr, ctx: FrozenBindings) -> Literal:
    if not e.expr:
        raise SPARQLError('Nothing given to cast.')
    if len(e.expr) > 1:
        raise SPARQLError('Cannot cast more than one thing!')
    x = e.expr[0]
    if e.iri == XSD.string:
        if isinstance(x, (URIRef, Literal)):
            return Literal(x, datatype=XSD.string)
        else:
            raise SPARQLError('Cannot cast term %r of type %r' % (x, type(x)))
    if not isinstance(x, Literal):
        raise SPARQLError('Can only cast Literals to non-string data-types')
    if x.datatype and (not x.datatype in XSD_DTs):
        raise SPARQLError('Cannot cast literal with unknown datatype: %r' % x.datatype)
    if e.iri == XSD.dateTime:
        if x.datatype and x.datatype not in (XSD.dateTime, XSD.string):
            raise SPARQLError('Cannot cast %r to XSD:dateTime' % x.datatype)
        try:
            return Literal(isodate.parse_datetime(x), datatype=e.iri)
        except:
            raise SPARQLError("Cannot interpret '%r' as datetime" % x)
    if x.datatype == XSD.dateTime:
        raise SPARQLError('Cannot cast XSD.dateTime to %r' % e.iri)
    if e.iri in (XSD.float, XSD.double):
        try:
            return Literal(float(x), datatype=e.iri)
        except:
            raise SPARQLError("Cannot interpret '%r' as float" % x)
    elif e.iri == XSD.decimal:
        if 'e' in x or 'E' in x:
            raise SPARQLError("Cannot interpret '%r' as decimal" % x)
        try:
            return Literal(Decimal(x), datatype=e.iri)
        except:
            raise SPARQLError("Cannot interpret '%r' as decimal" % x)
    elif e.iri == XSD.integer:
        try:
            return Literal(int(x), datatype=XSD.integer)
        except:
            raise SPARQLError("Cannot interpret '%r' as int" % x)
    elif e.iri == XSD.boolean:
        if x.lower() in ('1', 'true'):
            return Literal(True)
        if x.lower() in ('0', 'false'):
            return Literal(False)
        raise SPARQLError("Cannot interpret '%r' as bool" % x)