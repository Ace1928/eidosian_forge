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
@method(function('serialize', nargs=(1, 2), sequence_types=('item()*', 'element(output:serialization-parameters)?', 'xs:string')))
def evaluate_serialize_function(self, context=None):
    if self.context is not None:
        context = self.context
    params = self.get_argument(context, index=1) if len(self) == 2 else None
    kwargs = get_serialization_params(params, token=self)
    if context is None:
        raise self.missing_context()
    elif isinstance(context, XPathSchemaContext):
        return []
    method_ = kwargs.get('method', 'xml')
    if method_ in ('xml', 'html', 'text'):
        etree_module = context.etree
        if context.namespaces:
            for pfx, uri in context.namespaces.items():
                etree_module.register_namespace(pfx, uri)
        else:
            for pfx, uri in self.parser.namespaces.items():
                etree_module.register_namespace(pfx, uri)
        return serialize_to_xml(self[0].select(context), etree_module, **kwargs)
    elif method_ == 'json':
        return serialize_to_json(self[0].select(context), token=self, **kwargs)
    else:
        return []