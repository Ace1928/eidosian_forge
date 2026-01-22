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
@method(function('remove', nargs=2, sequence_types=('item()*', 'xs:integer', 'item()*')))
def select_remove_function(self, context=None):
    if self.context is not None:
        context = self.context
    position = self.get_argument(context, 1)
    if not isinstance(position, int):
        raise self.error('XPTY0004', 'an xs:integer required')
    for pos, result in enumerate(self[0].select(context), start=1):
        if pos != position:
            yield result