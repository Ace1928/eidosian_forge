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
def handle_microformat(root_class_names, el, value_property=None, simple_value=None, backcompat_mode=False):
    """Handles a (possibly nested) microformat, i.e. h-*"""
    properties = {}
    children = []
    self._default_date = None
    parsed_types_aggregation = set()
    if backcompat_mode:
        el = backcompat.apply_rules(el, self.__html_parser__, self.filtered_roots)
        root_class_names = mf2_classes.root(el.get('class', []), self.filtered_roots)
    root_lang = el.attrs.get('lang')
    for child in get_children(el):
        child_props, child_children, child_parsed_types_aggregation = parse_props(child, root_lang)
        for key, new_value in child_props.items():
            prop_value = properties.get(key, [])
            prop_value.extend(new_value)
            properties[key] = prop_value
        children.extend(child_children)
        parsed_types_aggregation.update(child_parsed_types_aggregation)
    if value_property and value_property in properties:
        simple_value = properties[value_property][0]
    if not backcompat_mode:
        if 'name' not in properties and parsed_types_aggregation.isdisjoint('peh'):
            properties['name'] = [implied_properties.name(el, self.__url__, self.filtered_roots)]
        if 'photo' not in properties and parsed_types_aggregation.isdisjoint('uh'):
            x = implied_properties.photo(el, self.__url__, self.filtered_roots)
            if x is not None:
                properties['photo'] = [x]
        if 'url' not in properties and parsed_types_aggregation.isdisjoint('uh'):
            x = implied_properties.url(el, self.__url__, self.filtered_roots)
            if x is not None:
                properties['url'] = [x]
    microformat = {'type': [class_name for class_name in sorted(root_class_names)], 'properties': properties}
    if el.name == 'area':
        shape = get_attr(el, 'shape')
        if shape is not None:
            microformat['shape'] = shape
        coords = get_attr(el, 'coords')
        if coords is not None:
            microformat['coords'] = coords
    if children:
        microformat['children'] = children
    Id = get_attr(el, 'id')
    if Id:
        microformat['id'] = Id
    if simple_value is not None:
        if isinstance(simple_value, dict):
            microformat.update(simple_value)
        else:
            microformat['value'] = simple_value
    if root_lang:
        microformat['lang'] = root_lang
    elif self.lang:
        microformat['lang'] = self.lang
    return microformat