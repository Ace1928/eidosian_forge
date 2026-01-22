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
@method(function('path', nargs=(0, 1), sequence_types=('node()?', 'xs:string?')))
def evaluate_path_function(self, context=None):
    if self.context is not None:
        context = self.context
    elif context is None:
        raise self.missing_context()
    if isinstance(context, XPathSchemaContext):
        return []
    elif not self:
        item = context.item
    else:
        item = self.get_argument(context)
        if item is None:
            return []
    suffix = ''
    if isinstance(item, DocumentNode):
        return '/'
    elif isinstance(item, (ElementNode, CommentNode, ProcessingInstructionNode)):
        elem = item.elem
    elif isinstance(item, TextNode):
        elem = item.parent.elem
        suffix = '/text()[1]'
    elif isinstance(item, AttributeNode):
        elem = item.parent.elem
        if item.name.startswith('{'):
            suffix = f'/@Q{item.name}'
        else:
            suffix = f'/@{item.name}'
    elif isinstance(item, NamespaceNode):
        elem = item.parent.elem
        if item.prefix:
            suffix = f'/namespace::{item.prefix}'
        else:
            suffix = f'/namespace::*[Q{{{XPATH_FUNCTIONS_NAMESPACE}}}local-name()=""]'
    else:
        return []
    if isinstance(context.root, DocumentNode):
        root = context.root.getroot().elem
        if root.tag.startswith('{'):
            path = f'/Q{root.tag}[1]'
        else:
            path = f'/Q{{}}{root.tag}[1]'
    else:
        root = context.root.elem
        path = 'Q{%s}root()' % XPATH_FUNCTIONS_NAMESPACE
    if isinstance(item, ProcessingInstructionNode):
        if item.parent is None or isinstance(item.parent, DocumentNode):
            return f'/processing-instruction({item.name})[{context.position}]'
    elif isinstance(item, CommentNode):
        if item.parent is None or isinstance(item.parent, DocumentNode):
            return f'/comment()[{context.position}]'
    for e, path in etree_iter_paths(root, path):
        if e is elem:
            return path + suffix
    else:
        return []