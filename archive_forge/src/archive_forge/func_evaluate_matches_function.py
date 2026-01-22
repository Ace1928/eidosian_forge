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
@method(function('matches', nargs=(2, 3), sequence_types=('xs:string?', 'xs:string', 'xs:string', 'xs:boolean')))
def evaluate_matches_function(self, context=None):
    if self.context is not None:
        context = self.context
    input_string = self.get_argument(context, default='', cls=str)
    pattern = self.get_argument(context, 1, required=True, cls=str)
    flags = 0
    if len(self) > 2:
        for c in self.get_argument(context, 2, required=True, cls=str):
            if c in 'smix':
                flags |= getattr(re, c.upper())
            elif c == 'q' and self.parser.version > '2':
                pattern = re.escape(pattern)
            else:
                raise self.error('FORX0001', 'Invalid regular expression flag %r' % c)
    try:
        python_pattern = translate_pattern(pattern, flags, self.parser.xsd_version)
        return re.search(python_pattern, input_string, flags=flags) is not None
    except (re.error, RegexError) as err:
        if isinstance(context, XPathSchemaContext):
            return False
        msg = 'Invalid regular expression: {}'
        raise self.error('FORX0002', msg.format(str(err))) from None
    except OverflowError as err:
        if isinstance(context, XPathSchemaContext):
            return False
        raise self.error('FORX0002', err) from None