import decimal
import math
from copy import copy
from decimal import Decimal
from itertools import product
from typing import TYPE_CHECKING, cast, Dict, Optional, List, Tuple, \
import urllib.parse
from .exceptions import ElementPathError, ElementPathValueError, \
from .helpers import ordinal, get_double, split_function_test
from .etree import is_etree_element, is_etree_document
from .namespaces import XSD_NAMESPACE, XPATH_FUNCTIONS_NAMESPACE, \
from .tree_builders import get_node_tree
from .xpath_nodes import XPathNode, ElementNode, AttributeNode, \
from .datatypes import xsd10_atomic_types, AbstractDateTime, AnyURI, \
from .protocols import ElementProtocol, DocumentProtocol, XsdAttributeProtocol, \
from .sequence_types import is_sequence_type_restriction, match_sequence_type
from .schema_proxy import AbstractSchemaProxy
from .tdop import Token, MultiLabel
from .xpath_context import XPathContext, XPathSchemaContext
def match_function_test(self, function_test: Union[str, List[Any]], as_argument: bool=False) -> bool:
    if isinstance(function_test, list):
        sequence_types = function_test
    else:
        sequence_types = split_function_test(function_test)
    if not sequence_types or not sequence_types[-1]:
        return False
    elif sequence_types[0] == '*':
        return True
    elif len(sequence_types) != 2:
        return False
    index_type, value_type = sequence_types
    if index_type.endswith(('+', '*')):
        return False
    return match_sequence_type(1, index_type) and all((match_sequence_type(v, value_type, self.parser) for v in self.items()))