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
def get_typed_node(self, item: PrincipalNodeType) -> PrincipalNodeType:
    """
        Returns a typed node if the item is matching an XSD type.

        Ref:
          https://www.w3.org/TR/xpath20/#id-processing-model
          https://www.w3.org/TR/xpath20/#id-static-analysis
          https://www.w3.org/TR/xquery-semantics/

        :param item: an untyped attribute or element.
        :return: a typed AttributeNode/ElementNode if the argument is matching         any associated XSD type.
        """
    if isinstance(item, (ElementNode, AttributeNode)) and item.xsd_type is not None:
        return item
    xsd_type = self.get_xsd_type(item)
    if xsd_type is not None and isinstance(item, (ElementNode, AttributeNode)):
        item.xsd_type = xsd_type
    return item