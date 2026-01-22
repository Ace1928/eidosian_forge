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
@method(function('xml-to-json', nargs=(1, 2), sequence_types=('node()?', 'map(*)', 'xs:string?')))
def evaluate_xml_to_json_function(self, context=None):
    if self.context is not None:
        context = self.context
    input_node = self.get_argument(context, cls=XPathNode)
    if input_node is None:
        return []
    if len(self) > 1:
        options = self.get_argument(context, index=1, required=True, cls=XPathMap)
        indent = options(context, 'indent')
        if indent is not None and isinstance(indent, bool):
            raise self.error('FOJS0005')

    def elem_to_json(elements):
        chunks = []

        def check_attributes(*exclude):
            for name in child.attrib:
                if name in exclude:
                    continue
                elif name.startswith('{') and (not name.startswith(f'{{{XPATH_FUNCTIONS_NAMESPACE}}}')):
                    continue
                raise self.error('FOJS0006', f'{child} has an invalid attribute {name!r}')

        def check_escapes(s):
            if re.search('(?<!\\\\)\\\\(?![urtnfb/"\\\\])', s):
                raise self.error('FOJS0007', f'invalid escape sequence in {s!r}')
            hex_digits = '0123456789abcdefABCDEF'
            for chunk in s.split('\\u')[1:]:
                if len(chunk) < 4 or any((x not in hex_digits for x in chunk[:4])):
                    raise self.error('FOJS0007', f'invalid unicode escape in {s!r}')
        for child in elements:
            if callable(child.tag):
                continue
            if child.tag == NULL_TAG:
                check_attributes()
                if child.text is not None:
                    msg = 'a null element cannot have a text value'
                    raise self.error('FOJS0006', msg)
                chunks.append('null')
            elif child.tag == BOOLEAN_TAG:
                check_attributes('key')
                if BooleanProxy(''.join(etree_iter_strings(child))):
                    chunks.append('true')
                else:
                    chunks.append('false')
            elif child.tag == NUMBER_TAG:
                check_attributes('key')
                value = ''.join(etree_iter_strings(child))
                try:
                    if self.parser.xsd_version == '1.0':
                        number = DoubleProxy10(value)
                    else:
                        number = DoubleProxy(value)
                except ValueError:
                    chunks.append('nan')
                else:
                    if math.isnan(number) or math.isinf(number):
                        msg = f'invalid number value {value!r}'
                        raise self.error('FOJS0006', msg)
                    chunks.append(str(number).rstrip('0').rstrip('.'))
            elif child.tag == STRING_TAG:
                check_attributes('key', 'escaped-key', 'escaped')
                if len(child):
                    msg = f'{child} cannot have element children'
                    raise self.error('FOJS0006', msg)
                value = ''.join(etree_iter_strings(child))
                check_escapes(value)
                escaped = child.get('escaped', '0').strip()
                if escaped not in BOOLEAN_VALUES:
                    msg = f"{child} has an invalid value for 'escaped' attribute"
                    raise self.error('FOJS0006', msg)
                value = escape_json_string(value, escaped in ('true', '1'))
                chunks.append(f'"{value}"')
            elif child.tag == ARRAY_TAG:
                check_attributes('key')
                if len(child):
                    if child.text is not None and child.text.strip() or any((e.tail and e.tail.strip() for e in child)):
                        msg = f'{child} has an invalid mixed content'
                        raise self.error('FOJS0006', msg)
                chunks.append(f'[{elem_to_json(child)}]')
            elif child.tag == MAP_TAG:
                map_chunks = []
                map_keys = set()
                for e in child:
                    key = e.get('key')
                    if not isinstance(key, str):
                        msg = f'object invalid key type {type(key)}'
                        raise self.error('FOJS0006', msg)
                    check_escapes(key)
                    escaped_key = e.get('escaped-key', '0').strip()
                    if escaped_key not in BOOLEAN_VALUES:
                        msg = f"{e} has an invalid value for 'escaped-key' attribute"
                        raise self.error('FOJS0006', msg)
                    key = escape_json_string(key, escaped=escaped_key in ('true', '1'))
                    map_chunks.append(f'"{key}":{elem_to_json((e,))}')
                    unescaped_key = unescape_json_string(key)
                    if unescaped_key in map_keys:
                        msg = f'key {key!r} duplication in map after escaping'
                        raise self.error('FOJS0006', msg)
                    map_keys.add(unescaped_key)
                chunks.append('{%s}' % ','.join(map_chunks))
            else:
                msg = f'invalid element tag {child.tag!r}'
                raise self.error('FOJS0006', msg)
        return ','.join(chunks)
    if isinstance(input_node, DocumentNode):
        return elem_to_json((child.value for child in input_node))
    elif isinstance(input_node, ElementNode):
        return elem_to_json((input_node.value,))
    else:
        raise self.error('FOJS0006')