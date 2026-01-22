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
@method(function('root', nargs=(0, 1), sequence_types=('node()?', 'node()?')))
def evaluate_root_function(self, context=None):
    if self.context is not None:
        context = self.context
    elif context is None:
        raise self.missing_context()
    if isinstance(context, XPathSchemaContext):
        return []
    elif not self:
        if not isinstance(context.item, XPathNode):
            raise self.error('XPTY0004')
        root = context.get_root(context.item)
        return root if root is not None else []
    else:
        item = self.get_argument(context)
        if item is None:
            return []
        elif not isinstance(item, XPathNode):
            raise self.error('XPTY0004')
        root = context.get_root(item)
        return root if root is not None else []