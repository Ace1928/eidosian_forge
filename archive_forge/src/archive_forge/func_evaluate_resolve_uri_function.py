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
@method(function('resolve-uri', nargs=(1, 2), sequence_types=('xs:string?', 'xs:string', 'xs:anyURI?')))
def evaluate_resolve_uri_function(self, context=None):
    if self.context is not None:
        context = self.context
    relative = self.get_argument(context, cls=str)
    if len(self) == 1:
        if self.parser.base_uri is None:
            raise self.error('FONS0005')
        elif relative is None:
            return []
        elif not AnyURI.is_valid(relative):
            raise self.error('FORG0002', '{!r} is not a valid URI'.format(relative))
        else:
            return self.get_absolute_uri(relative, as_string=False)
    base_uri = self.get_argument(context, index=1, required=True, cls=str)
    if not AnyURI.is_valid(base_uri):
        raise self.error('FORG0002', '{!r} is not a valid URI'.format(base_uri))
    elif relative is None:
        return []
    elif not AnyURI.is_valid(relative):
        raise self.error('FORG0002', '{!r} is not a valid URI'.format(relative))
    else:
        return self.get_absolute_uri(relative, base_uri, as_string=False)