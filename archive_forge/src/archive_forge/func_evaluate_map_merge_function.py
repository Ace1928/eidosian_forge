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
@method(function('merge', prefix='map', nargs=(1, 2), sequence_types=('map(*)*', 'map(*)', 'map(*)')))
def evaluate_map_merge_function(self, context=None):
    if self.context is not None:
        context = self.context
    duplicates = 'use-first'
    if len(self) > 1:
        options = self.get_argument(context, index=1, required=True, cls=XPathMap)
        for opt, value in options.items(context):
            if opt == 'duplicates':
                if value in ('reject', 'use-first', 'use-last', 'use-any', 'combine'):
                    duplicates = value
                else:
                    raise self.error('FOJS0005')
    items = {}
    for map_ in self[0].select(context):
        for k1, v in map_.items(context):
            if isinstance(k1, SAFE_KEY_ATOMIC_TYPES) or (isinstance(k1, float) and (not math.isnan(k1))):
                if k1 not in items:
                    items[k1] = v
                elif duplicates == 'reject':
                    raise self.error('FOJS0003')
                elif duplicates == 'use-last':
                    items.pop(k1)
                    items[k1] = v
                elif duplicates == 'combine':
                    try:
                        items[k1].append(v)
                    except AttributeError:
                        items[k1] = [items[k1], v]
                continue
            for k2 in items:
                if same_key(k1, k2):
                    if duplicates == 'reject':
                        raise self.error('FOJS0003')
                    elif duplicates == 'use-last':
                        items.pop(k2)
                        items[k1] = v
                    elif duplicates == 'combine':
                        try:
                            items[k2].append(v)
                        except AttributeError:
                            items[k2] = [items[k2], v]
                    break
            else:
                items[k1] = v
    return XPathMap(self.parser, items)