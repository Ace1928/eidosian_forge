import decimal
import os
import re
import codecs
import math
from copy import copy
from itertools import zip_longest
from typing import cast, Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit
from urllib.request import urlopen
from urllib.error import URLError
from ..exceptions import ElementPathError
from ..tdop import MultiLabel
from ..helpers import OCCURRENCE_INDICATORS, EQNAME_PATTERN, \
from ..namespaces import get_expanded_name, split_expanded_name, \
from ..datatypes import xsd10_atomic_types, NumericProxy, QName, Date10, \
from ..sequence_types import is_sequence_type, match_sequence_type
from ..etree import defuse_xml, etree_iter_paths
from ..xpath_nodes import XPathNode, ElementNode, TextNode, AttributeNode, \
from ..tree_builders import get_node_tree
from ..xpath_tokens import XPathFunctionArgType, XPathToken, ValueToken, XPathFunction
from ..serialization import get_serialization_params, serialize_to_xml, serialize_to_json
from ..xpath_context import XPathContext, XPathSchemaContext
from ..regex import translate_pattern, RegexError
from ._xpath30_operators import XPath30Parser
from .xpath30_helpers import UNICODE_DIGIT_PATTERN, DECIMAL_DIGIT_PATTERN, \
@method(function('outermost', nargs=1, sequence_types=('node()*', 'node()*')))
def select_outermost_function(self, context=None):
    if context is None:
        raise self.missing_context()
    context = copy(context)
    nodes = [e for e in self[0].select(context)]
    if any((not isinstance(x, XPathNode) for x in nodes)):
        raise self.error('XPTY0004', 'argument must contain only nodes')
    results = set()
    if len(nodes) > 10:
        nodes = set(nodes)
    for item in nodes:
        context.item = item
        ancestors = {x for x in context.iter_ancestors(axis='ancestor')}
        if any((x in nodes for x in ancestors)):
            continue
        results.add(item)
    yield from sorted(results, key=node_position)