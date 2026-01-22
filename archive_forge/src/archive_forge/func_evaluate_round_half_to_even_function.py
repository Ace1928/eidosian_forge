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
@method(function('round-half-to-even', nargs=(1, 2), sequence_types=('xs:numeric?', 'xs:integer', 'xs:numeric?')))
def evaluate_round_half_to_even_function(self, context=None):
    if self.context is not None:
        context = self.context
    item = self.get_argument(context)
    if item is None:
        return []
    elif isinstance(item, float) and (math.isnan(item) or math.isinf(item)):
        return item
    elif not isinstance(item, (float, int, Decimal)):
        code = 'XPTY0004' if isinstance(item, str) else 'FORG0006'
        raise self.error(code, 'invalid argument type {!r}'.format(type(item)))
    precision = 0 if len(self) < 2 else self[1].evaluate(context)
    try:
        if isinstance(item, int):
            return round(item, precision)
        elif isinstance(item, Decimal):
            return round(item, precision)
        elif isinstance(item, Float10):
            return Float10(round(item, precision))
        return float(round(Decimal.from_float(item), precision))
    except TypeError as err:
        if isinstance(context, XPathSchemaContext):
            return []
        raise self.error('XPTY0004', err)
    except (DecimalException, OverflowError):
        if isinstance(item, Decimal):
            return Decimal.from_float(round(float(item), precision))
        return round(item, precision)