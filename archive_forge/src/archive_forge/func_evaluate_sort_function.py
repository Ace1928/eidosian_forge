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
@method(function('sort', nargs=(1, 3), sequence_types=('item()*', 'xs:string?', 'function(item()) as xs:anyAtomicType*', 'item()*')))
def evaluate_sort_function(self, context=None):
    if self.context is not None:
        context = self.context
    if len(self) < 2:
        collation = self.parser.default_collation
    else:
        collation = self.get_argument(context, 1, cls=str)
        if collation is None:
            collation = self.parser.default_collation
    if len(self) == 3:
        func = self.get_argument(context, index=2, required=True, cls=XPathFunction)
        key_function = get_key_function(collation, key_func=lambda x: func(x, context=context), token=self)
    else:
        key_function = get_key_function(collation, token=self)
    try:
        return sorted(self[0].select(context), key=key_function)
    except ElementPathTypeError:
        raise
    except TypeError:
        if isinstance(context, XPathSchemaContext):
            return []
        raise self.error('XPTY0004')