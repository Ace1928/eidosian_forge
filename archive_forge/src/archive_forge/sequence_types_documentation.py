import re
from itertools import zip_longest
from typing import TYPE_CHECKING, cast, Any, Optional
from .exceptions import ElementPathKeyError, xpath_error
from .helpers import OCCURRENCE_INDICATORS, EQNAME_PATTERN, WHITESPACES_PATTERN
from .namespaces import XSD_NAMESPACE, XSD_ERROR, XSD_ANY_SIMPLE_TYPE, XSD_NUMERIC, \
from .datatypes import xsd10_atomic_types, xsd11_atomic_types, AnyAtomicType, \
from .xpath_nodes import XPathNode, DocumentNode, ElementNode, AttributeNode
from . import xpath_tokens

    Checks a value instance against a sequence type.

    :param value: the instance to check.
    :param sequence_type: a string containing the sequence type spec.
    :param parser: an optional parser instance for type checking.
    :param strict: if `False` match xs:anyURI with strings.
    