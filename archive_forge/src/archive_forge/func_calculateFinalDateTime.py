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
def calculateFinalDateTime(obj1: Union[py_datetime.date, py_datetime.datetime], dt1: URIRef, obj2: Union[isodate.Duration, py_datetime.timedelta], dt2: URIRef, operation: str) -> Literal:
    """
    Calculates the final dateTime/date/time resultant after addition/
    subtraction of duration/dayTimeDuration/yearMonthDuration
    """
    if isCompatibleDateTimeDatatype(obj1, dt1, obj2, dt2):
        if operation == '-':
            ans = obj1 - obj2
            return Literal(ans, datatype=dt1)
        else:
            ans = obj1 + obj2
            return Literal(ans, datatype=dt1)
    else:
        raise SPARQLError('Incompatible Data types to DateTime Operations')