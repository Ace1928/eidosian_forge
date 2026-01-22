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
@method(function('json-to-xml', nargs=(1, 2), sequence_types=('xs:string?', 'map(*)', 'document-node()?')))
def evaluate_json_to_xml_function(self, context=None):
    if self.context is not None:
        context = self.context
    json_text = self.get_argument(context, cls=str)
    if json_text is None or isinstance(context, XPathSchemaContext):
        return []
    elif context is not None:
        etree = context.etree
    else:
        raise self.missing_context()

    def _fallback(*_args, **_kwargs):
        return '&#xFFFD;'
    liberal = False
    validate = False
    duplicates = None
    escape = False
    fallback = _fallback
    if len(self) > 1:
        options = self.get_argument(context, index=1, required=True, cls=XPathMap)
        for key, value in options.items(context):
            if key == 'liberal':
                if not isinstance(value, bool):
                    raise self.error('XPTY0004')
                liberal = value
            elif key == 'duplicates':
                if not isinstance(value, str):
                    raise self.error('XPTY0004')
                elif value not in ('reject', 'retain', 'use-first'):
                    raise self.error('FOJS0005')
                duplicates = value
            elif key == 'validate':
                if not isinstance(value, bool):
                    raise self.error('XPTY0004')
                validate = value
            elif key == 'escape':
                if not isinstance(value, bool):
                    raise self.error('XPTY0004')
                escape = value
            elif key == 'fallback':
                if escape:
                    msg = "'fallback' function provided with escape=True"
                    raise self.error('FOJS0005', msg)
                if not isinstance(value, XPathFunction):
                    raise self.error('XPTY0004')
                fallback = value
            else:
                raise self.error('FOJS0005')
        if duplicates is None:
            duplicates = 'reject' if validate else 'retain'
        elif validate and duplicates == 'retain':
            raise self.error('FOJS0005')

    def escape_string(s):
        s = re.sub('\\\\(?!/)', '\\\\\\\\', s)
        s = s.replace('\x08', '\\b').replace('\r', '\\r').replace('\n', '\\n').replace('\t', '\\t').replace('\x0c', '\\f').replace('/', '\\/')
        return ''.join((x if is_xml_codepoint(ord(x)) else f'\\u{ord(x):04X}' for x in s))

    def value_to_etree(v, **attrib):
        if v is None:
            elem = etree.Element(NULL_TAG, **attrib)
        elif isinstance(v, list):
            elem = etree.Element(ARRAY_TAG, **attrib)
            for item in v:
                elem.append(value_to_etree(item))
        elif isinstance(v, bool):
            elem = etree.Element(BOOLEAN_TAG, **attrib)
            elem.text = 'true' if v else 'false'
        elif isinstance(v, (int, float)):
            elem = etree.Element(NUMBER_TAG, **attrib)
            elem.text = str(v)
        elif isinstance(v, str):
            if not escape:
                v = ''.join((x if is_xml_codepoint(ord(x)) else fallback(f'\\u{ord(x):04X}', context=context) for x in v))
                elem = etree.Element(STRING_TAG, **attrib)
            else:
                v = escape_string(v)
                if '\\' in v:
                    elem = etree.Element(STRING_TAG, escaped='true', **attrib)
                else:
                    elem = etree.Element(STRING_TAG, **attrib)
            elem.text = v
        elif is_etree_element(v):
            v.attrib.update(attrib)
            return v
        else:
            raise ElementPathTypeError(f'unexpected type {type(v)}')
        return elem

    def json_object_to_etree(obj):
        keys = set()
        items = []
        for k, v in obj:
            if k not in keys:
                keys.add(k)
            elif duplicates == 'use-first':
                continue
            elif duplicates == 'reject':
                raise self.error('FOJS0003')
            if not escape:
                k = ''.join((x if is_xml_codepoint(ord(x)) else fallback(f'\\u{ord(x):04X}', context=context) for x in k))
                k = k.replace('"', '&#34;')
                attrib = {'key': k}
            else:
                k = escape_string(k)
                if '\\' in k:
                    attrib = {'escaped-key': 'true', 'key': k}
                else:
                    attrib = {'key': k}
            items.append(value_to_etree(v, **attrib))
        elem = etree.Element(MAP_TAG)
        for item in items:
            elem.append(item)
        return elem
    kwargs = {'object_pairs_hook': json_object_to_etree}
    if liberal or escape:
        kwargs['strict'] = False
    if liberal:

        def parse_constant(s):
            raise self.error('FOJS0001')
        kwargs['parse_constant'] = parse_constant
    etree.register_namespace('fn', XPATH_FUNCTIONS_NAMESPACE)
    try:
        if json_text.startswith('\ufeff'):
            result = json.JSONDecoder(**kwargs).decode(json_text[1:])
        else:
            result = json.JSONDecoder(**kwargs).decode(json_text)
    except json.JSONDecodeError as err:
        raise self.error('FOJS0001', str(err)) from None
    if is_etree_element(result):
        document = etree.ElementTree(result)
    else:
        document = etree.ElementTree(value_to_etree(result))
    root = document.getroot()
    if XML_BASE not in root.attrib and self.parser.base_uri:
        root.set(XML_BASE, self.parser.base_uri)
    if validate:
        validate_json_to_xml(document.getroot())
    return get_node_tree(document, namespaces={'j': XPATH_FUNCTIONS_NAMESPACE})