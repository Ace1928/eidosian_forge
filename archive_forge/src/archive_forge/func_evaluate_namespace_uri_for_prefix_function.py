import math
import datetime
import time
import re
import os.path
import unicodedata
from copy import copy
from decimal import Decimal, DecimalException
from string import ascii_letters
from urllib.parse import urlsplit, quote as urllib_quote
from ..exceptions import ElementPathValueError
from ..helpers import QNAME_PATTERN, is_idrefs, is_xml_codepoint, round_number
from ..datatypes import DateTime10, DateTime, Date10, Date, Float10, \
from ..namespaces import XML_NAMESPACE, get_namespace, split_expanded_name, \
from ..compare import deep_equal
from ..sequence_types import match_sequence_type
from ..xpath_context import XPathSchemaContext
from ..xpath_nodes import XPathNode, DocumentNode, ElementNode, SchemaElementNode
from ..xpath_tokens import XPathFunction
from ..regex import RegexError, translate_pattern
from ..collations import CollationManager
from ._xpath2_operators import XPath2Parser
@method(function('namespace-uri-for-prefix', nargs=2, sequence_types=('xs:string?', 'element()', 'xs:anyURI?')))
def evaluate_namespace_uri_for_prefix_function(self, context=None):
    if self.context is not None:
        context = self.context
    elif context is None:
        raise self.missing_context()
    prefix = self.get_argument(context=copy(context))
    if prefix is None:
        prefix = ''
    if not isinstance(prefix, str):
        raise self.error('FORG0006', '1st argument has an invalid type %r' % type(prefix))
    elem = self.get_argument(context, index=1)
    if not isinstance(elem, ElementNode):
        raise self.error('FORG0006', '2nd argument %r is not an element node' % elem)
    ns_uris = {get_namespace(e.tag) for e in elem.elem.iter() if not callable(e.tag)}
    for p, uri in self.parser.namespaces.items():
        if uri in ns_uris:
            if p == prefix:
                if not prefix or uri:
                    return AnyURI(uri)
                else:
                    msg = 'Prefix %r is associated to no namespace'
                    raise self.error('XPST0081', msg % prefix)
    else:
        return []