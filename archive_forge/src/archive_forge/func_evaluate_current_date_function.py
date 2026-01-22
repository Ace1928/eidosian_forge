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
@method(function('current-date', nargs=0, sequence_types=('xs:date',)))
def evaluate_current_date_function(self, context=None):
    if self.context is not None:
        context = self.context
    dt = datetime.datetime.now() if context is None else context.current_dt
    if self.parser.xsd_version == '1.1':
        return Date(dt.year, dt.month, dt.day, tzinfo=dt.tzinfo)
    return Date10(dt.year, dt.month, dt.day, tzinfo=dt.tzinfo)