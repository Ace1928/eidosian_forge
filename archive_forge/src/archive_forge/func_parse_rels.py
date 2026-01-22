import copy
import json
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup, FeatureNotFound
from bs4.element import Tag
from . import (
from .dom_helpers import get_attr, get_children, get_descendents, try_urljoin
from .mf_helpers import unordered_list
from .version import __version__
def parse_rels(el):
    """Parse an element for rel microformats"""
    rel_attrs = get_attr(el, 'rel')
    if rel_attrs is not None:
        url = try_urljoin(self.__url__, el.get('href', ''))
        value_dict = self.__parsed__['rel-urls'].get(url, {})
        if 'text' not in value_dict:
            value_dict['text'] = el.get_text().strip()
        url_rels = value_dict.get('rels', [])
        value_dict['rels'] = url_rels
        for knownattr in ('media', 'hreflang', 'type', 'title'):
            x = get_attr(el, knownattr)
            if x is not None and knownattr not in value_dict:
                value_dict[knownattr] = x
        self.__parsed__['rel-urls'][url] = value_dict
        for rel_value in rel_attrs:
            value_list = self.__parsed__['rels'].get(rel_value, [])
            if url not in value_list:
                value_list.append(url)
            if rel_value not in url_rels:
                url_rels.append(rel_value)
            self.__parsed__['rels'][rel_value] = value_list
        if 'alternate' in rel_attrs:
            alternate_list = self.__parsed__.get('alternates', [])
            alternate_dict = {}
            alternate_dict['url'] = url
            x = ' '.join([r for r in rel_attrs if not r == 'alternate'])
            if x != '':
                alternate_dict['rel'] = x
            alternate_dict['text'] = el.get_text().strip()
            for knownattr in ('media', 'hreflang', 'type', 'title'):
                x = get_attr(el, knownattr)
                if x is not None:
                    alternate_dict[knownattr] = x
            alternate_list.append(alternate_dict)
            self.__parsed__['alternates'] = alternate_list