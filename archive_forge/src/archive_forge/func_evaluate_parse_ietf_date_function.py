import json
import locale
import math
import pathlib
import random
import re
from datetime import datetime, timedelta
from decimal import Decimal
from itertools import product
from urllib.request import urlopen
from urllib.parse import urlsplit
from ..datatypes import AnyAtomicType, AbstractBinary, AbstractDateTime, \
from ..exceptions import ElementPathTypeError
from ..helpers import WHITESPACES_PATTERN, is_xml_codepoint, \
from ..namespaces import XPATH_FUNCTIONS_NAMESPACE, XML_BASE
from ..etree import etree_iter_strings, is_etree_element
from ..collations import CollationManager
from ..compare import get_key_function, same_key
from ..tree_builders import get_node_tree
from ..xpath_nodes import XPathNode, DocumentNode, ElementNode
from ..xpath_tokens import XPathFunction, XPathMap, XPathArray
from ..xpath_context import XPathSchemaContext
from ..validators import validate_json_to_xml
from ._xpath31_operators import XPath31Parser
@method(function('parse-ietf-date', nargs=1, sequence_types=('xs:string?', 'xs:dateTime?')))
def evaluate_parse_ietf_date_function(self, context=None):
    if self.context is not None:
        context = self.context
    value = self.get_argument(context, cls=str)
    if value is None:
        return []
    value = WHITESPACES_PATTERN.sub(' ', value).strip()
    value = value.replace(' -', '-').replace('- ', '-').replace(' +', '+')
    value = value.replace(' (', '(').replace('( ', '(').replace(' )', ')')
    if re.search('(?<=[+\\-])(\\d{2}:\\d)(?=\\D)', value) is not None:
        raise self.error('FORG0010')
    if re.search(' \\d{1,2}:\\d(?=\\D)', value) is not None:
        raise self.error('FORG0010')
    value = re.sub('(?<=\\D)(\\d)(?=\\D)', '0\\g<1>', value)
    value = re.sub('(?<=\\d[+\\-])(\\d{2}:)(?=($|[ (]))', '\\g<1>00', value)
    value = re.sub('(?<=\\d[+\\-])(\\d{2})(?=($|[ (]))', '\\g<1>:00', value)
    value = re.sub('(?<=\\d[+\\-])(\\d{3})(?=[ (])', '0\\g<1>', value)
    tzname_regex = '(?<=[\\d( ])(UT|UTC|GMT|EST|EDT|CST|CDT|MST|MDT|PST|PDT)\\b'
    tzname_match = re.search(tzname_regex, value, re.IGNORECASE)
    if tzname_match is not None:
        value = re.sub(tzname_regex, 'UTC', value, flags=re.IGNORECASE)
    illegal_tzname_regex = '\\b(CET)\\b'
    if re.search(illegal_tzname_regex, value, re.IGNORECASE) is not None:
        raise self.error('FORG0010', 'illegal timezone name')
    if value and value[0].isalpha():
        try:
            dayname, _value = value.split(' ', maxsplit=1)
        except ValueError:
            raise self.error('FORG0010') from None
        else:
            if dayname.endswith(','):
                dayname = dayname[:-1]
            for fmt in ['%A', '%a']:
                try:
                    datetime.strptime(dayname, fmt)
                except ValueError:
                    pass
                else:
                    value = _value
                    break
    if ' 24:00 ' in value:
        value = value.replace(' 24:00 ', ' 00:00 ')
        day_offset = True
    elif ' 24:00:00' in value and ' 24:00:00.' not in value:
        value = value.replace(' 24:00:00', ' 00:00:00')
        day_offset = True
    else:
        day_offset = False
    if value and value[0].isalpha():
        fmt_alternatives = (['%b %d %H:%M', '%b-%d %H:%M'], ['', ':%S', ':%S.%f'], ['', '%Z', ' %Z', '%z', '%z(%Z)'], [' %Y', ' %y'])
        value = re.sub('(?<= )(\\d{2})$', '19\\g<1>', value)
    else:
        fmt_alternatives = (['%d %b ', '%d-%b-', '%d %b-', '%d-%b '], ['%Y %H:%M', '%y %H:%M'], ['', ':%S', ':%S.%f'], ['', '%Z', ' %Z', '%z', '%z(%Z)'])
        value = re.sub('(?<=[ \\-])(\\d{2})(?= \\d{2}:\\d{2})', '19\\g<1>', value)
    for fmt_chunks in product(*fmt_alternatives):
        fmt = ''.join(fmt_chunks)
        if '%f%Z' in fmt:
            continue
        try:
            dt = datetime.strptime(value, fmt)
        except ValueError:
            continue
        else:
            if tzname_match is not None and dt.tzinfo is None:
                tzname = tzname_match.group(0).upper()
                dt = dt.replace(tzinfo=Timezone.fromstring(TIMEZONE_MAP[tzname]))
            if dt.tzinfo is not None:
                offset = dt.tzinfo.utcoffset(None)
                seconds = offset.days * 86400 + offset.seconds
                if abs(seconds) > 14 * 3600:
                    raise self.error('FORG0010')
            if day_offset:
                dt = dt + timedelta(seconds=86400)
            return DateTime.fromdatetime(dt)
    else:
        raise self.error('FORG0010')