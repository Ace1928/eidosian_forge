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
@method('format-dateTime')
@method('format-date')
@method('format-time')
def evaluate_format_date_time_functions(self, context=None):
    if self.symbol == 'format-dateTime':
        cls = DateTime10
        invalid_markers = ''
    elif self.symbol == 'format-date':
        cls = Date10
        invalid_markers = 'HhPmsf'
    else:
        cls = Time
        invalid_markers = 'YMDdFWwCE'
    if self.context is not None:
        context = self.context
    value = self.get_argument(context, cls=cls)
    picture = self.get_argument(context, index=1, required=True, cls=str)
    if len(self) not in [2, 5]:
        raise self.error('XPST0017')
    language = self.get_argument(context, index=2, cls=str)
    calendar = self.get_argument(context, index=3, cls=str)
    place = self.get_argument(context, index=4, cls=str)
    if value is None:
        return ''
    try:
        literals, markers = parse_datetime_picture(picture)
    except ElementPathError as err:
        err.token = self
        raise
    if invalid_markers:
        for mrk in markers:
            if mrk[1] in invalid_markers:
                msg = 'Invalid date formatting component {!r}'.format(mrk)
                raise self.error('FOFD1350', msg)
    result = []
    if language not in ('en', 'it', None):
        language = 'en'
        result.append('[Language: en')
    if calendar is not None:
        if calendar.startswith('Q{}'):
            calendar = calendar[3:]
        if calendar not in ('AD', 'ISO', 'OS'):
            if context is None or calendar != context.default_calendar:
                if QName.is_valid(calendar):
                    if ':' not in calendar:
                        msg = f'unknown calendar in no namespace {calendar!r}'
                        raise self.error('FOFD1340', msg)
                    try:
                        _ = get_expanded_name(calendar, self.parser.namespaces)
                    except (KeyError, ValueError) as err:
                        raise self.error('FOFD1340', str(err)) from None
                elif EQNAME_PATTERN.search(calendar) is None:
                    raise self.error('FOFD1340', f'Invalid calendar argument {calendar!r}')
            else:
                result.append('[' if not result else ', ')
                result.append('Calendar: AD')
    if place is not None and zoneinfo is not None:
        try:
            zone = zoneinfo.ZoneInfo(place.strip())
        except zoneinfo.ZoneInfoNotFoundError:
            if not isinstance(context, XPathSchemaContext):
                raise self.error('FOFD1340', f'Invalid place argument {place!r}')
        else:
            value = value.astimezone(zone)
    if result:
        result.append(']')
    for k in range(len(markers)):
        result.append(literals[k])
        try:
            result.append(parse_datetime_marker(markers[k], value, language))
        except ElementPathError as err:
            err.token = self
            raise
    result.append(literals[-1])
    return ''.join(result)