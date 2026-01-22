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
@method(function('error', nargs=(0, 3), sequence_types=('xs:QName?', 'xs:string', 'item()*', 'none')))
def evaluate_error_function(self, context=None):
    if self.context is not None:
        context = self.context
    if not self:
        raise self.error('FOER0000')
    elif len(self) == 1:
        error = self.get_argument(context, cls=QName)
        if error is None:
            raise self.error('XPTY0004', 'an xs:QName expected')
        raise self.error(error or 'FOER0000')
    else:
        error = self.get_argument(context, cls=QName)
        description = self.get_argument(context, index=1, cls=str)
        raise self.error(error or 'FOER0000', description)