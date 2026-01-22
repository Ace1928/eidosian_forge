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
@method(function('tokenize', nargs=(1, 3), sequence_types=('xs:string?', 'xs:string', 'xs:string', 'xs:string*')))
def evaluate_tokenize_function(self, context=None):
    if self.context is not None:
        context = self.context
    input_string = self.get_argument(context, cls=str)
    if input_string is None:
        return []
    elif self.parser.version >= '3.1' and len(self) == 1:
        pattern = ' '
        input_string = ' '.join(re.split('[ \t\n\r\x0c\x0b]+', input_string.strip(' \t\n\r\x0c\x0b')))
    else:
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
        pattern = re.compile(python_pattern, flags=flags)
    except (re.error, RegexError):
        if isinstance(context, XPathSchemaContext):
            return [input_string]
        raise self.error('FORX0002', 'Invalid regular expression %r' % pattern) from None
    else:
        if pattern.search(''):
            msg = 'Regular expression %r matches zero-length string'
            raise self.error('FORX0003', msg % pattern.pattern)
    result = []
    if input_string:
        for value in pattern.split(input_string):
            if value is not None and pattern.search(value) is None:
                result.append(value)
        if len(result) == 1:
            return result[0]
    return result