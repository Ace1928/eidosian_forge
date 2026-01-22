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
@method(function('format-integer', nargs=(2, 3), sequence_types=('xs:integer?', 'xs:string', 'xs:string?', 'xs:string')))
def evaluate_format_integer_function(self, context=None):
    if self.context is not None:
        context = self.context
    value = self.get_argument(context, cls=NumericProxy)
    picture = self.get_argument(context, index=1, required=True, cls=str)
    lang = self.get_argument(context, index=2, cls=str)
    if value is None:
        return ''
    if ';' not in picture:
        fmt_token, fmt_modifier = (picture, '')
    else:
        fmt_token, fmt_modifier = picture.rsplit(';', 1)
    if MODIFIER_PATTERN.match(fmt_modifier) is None:
        raise self.error('FODF1310')
    if not fmt_token:
        raise self.error('FODF1310')
    elif fmt_token in FORMAT_INTEGER_TOKENS:
        if fmt_token == 'a':
            result = int_to_alphabetic(value, lang)
        elif fmt_token == 'A':
            result = int_to_alphabetic(value, lang).upper()
        elif fmt_token == 'i':
            result = int_to_roman(value).lower()
        elif fmt_token == 'I':
            result = int_to_roman(value)
        elif fmt_token == 'w':
            return int_to_words(value, lang, fmt_modifier)
        elif fmt_token == 'W':
            return int_to_words(value, lang, fmt_modifier).upper()
        else:
            return int_to_words(value, lang, fmt_modifier).title()
    elif UNICODE_DIGIT_PATTERN.search(fmt_token) is None:
        if any((not x.isalpha() and (not x.isdigit()) for x in fmt_token)):
            result = str(value)
        else:
            base_char = '1'
            for base_char in fmt_token:
                if base_char.isalpha():
                    break
            if base_char.islower():
                result = int_to_alphabetic(value, base_char)
            else:
                result = int_to_alphabetic(value, base_char.lower()).upper()
    elif DECIMAL_DIGIT_PATTERN.search(fmt_token) is None or ',,' in fmt_token:
        msg = 'picture argument has an invalid primary format token'
        raise self.error('FODF1310', msg)
    else:
        digits = UNICODE_DIGIT_PATTERN.findall(fmt_token)
        cp = ord(digits[0])
        if any((ord(ch) - cp > 10 for ch in digits[1:])):
            msg = 'picture argument mixes digits from different digit families'
            raise self.error('FODF1310', msg)
        elif fmt_token[0].isdigit():
            if '#' in fmt_token:
                msg = 'picture argument has an invalid primary format token'
                raise self.error('FODF1310', msg)
        elif fmt_token[0] != '#':
            raise self.error('FODF1310', 'invalid grouping in picture argument')
        if digits[0].isdigit():
            cp = ord(digits[0])
            while chr(cp - 1).isdigit():
                cp -= 1
            digits_family = ''.join((chr(cp + k) for k in range(10)))
        else:
            raise ValueError()
        if value < 0:
            result = '-' + format_digits(str(abs(value)), fmt_token, digits_family)
        else:
            result = format_digits(str(abs(value)), fmt_token, digits_family)
    if fmt_modifier.startswith('o'):
        return f'{result}{ordinal_suffix(value)}'
    return result