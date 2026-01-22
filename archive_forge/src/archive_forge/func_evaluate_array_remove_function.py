import json
import locale
import math
import pathlib
import random
import re
from datetime import datetime, timedelta
from decimal import Decimal
from itertools import product
from urllib.request import urlopen
from urllib.parse import urlsplit
from ..datatypes import AnyAtomicType, AbstractBinary, AbstractDateTime, \
from ..exceptions import ElementPathTypeError
from ..helpers import WHITESPACES_PATTERN, is_xml_codepoint, \
from ..namespaces import XPATH_FUNCTIONS_NAMESPACE, XML_BASE
from ..etree import etree_iter_strings, is_etree_element
from ..collations import CollationManager
from ..compare import get_key_function, same_key
from ..tree_builders import get_node_tree
from ..xpath_nodes import XPathNode, DocumentNode, ElementNode
from ..xpath_tokens import XPathFunction, XPathMap, XPathArray
from ..xpath_context import XPathSchemaContext
from ..validators import validate_json_to_xml
from ._xpath31_operators import XPath31Parser
@method(function('remove', prefix='array', nargs=2, sequence_types=('array(*)', 'xs:integer*', 'array(*)')))
def evaluate_array_remove_function(self, context=None):
    if self.context is not None:
        context = self.context
    array_ = self.get_argument(context, required=True, cls=XPathArray)
    positions_ = self[1].evaluate(context)
    if positions_ is None:
        return array_
    positions = positions_ if isinstance(positions_, list) else [positions_]
    if any((p <= 0 or p > len(array_) for p in positions)):
        if isinstance(context, XPathSchemaContext):
            return array_
        raise self.error('FOAY0001')
    items = (v for k, v in enumerate(array_.items(context), 1) if k not in positions)
    return XPathArray(self.parser, items=items)