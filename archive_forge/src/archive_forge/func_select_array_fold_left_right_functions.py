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
@method(function('fold-left', prefix='array', nargs=3, sequence_types=('array(*)', 'item()*', 'function(item()*, item()) as item()*', 'item()*')))
@method(function('fold-right', prefix='array', nargs=3, sequence_types=('array(*)', 'item()*', 'function(item()*, item()) as item()*', 'item()*')))
def select_array_fold_left_right_functions(self, context=None):
    if self.context is not None:
        context = self.context
    func = self[2][1] if self[2].symbol == ':' else self[2]
    if not isinstance(func, XPathFunction):
        func = self.get_argument(context, index=2, cls=XPathFunction, required=True)
    if func.arity != 2:
        raise self.error('XPTY0004', 'function arity must be 2')
    array_ = self.get_argument(context, required=True, cls=XPathArray)
    zero = self.get_argument(context, index=1)
    result = zero
    if self.symbol == 'fold-left':
        for item in array_.items(context):
            result = func(result, item, context=context)
    else:
        for item in reversed(array_.items(context)):
            result = func(item, result, context=context)
    if isinstance(result, list):
        yield from result
    else:
        yield result