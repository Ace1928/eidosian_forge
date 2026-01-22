import re
from urllib.parse import urljoin
import bs4
from bs4.element import Comment, NavigableString, Tag
def text_collection(el, replace_img=False, img_to_src=True, base_url=''):
    items = []
    if el.name in DROP_TAGS or isinstance(el, Comment):
        items = []
    elif isinstance(el, NavigableString):
        value = el
        value = _whitespace_to_space_regex.sub(' ', value)
        items = [_reduce_spaces_regex.sub(' ', value)]
    elif el.name in PRE_TAGS:
        items = [PRE_BEFORE, el.get_text(), PRE_AFTER]
    elif el.name == 'img' and replace_img:
        value = el.get('alt')
        if value is None and img_to_src:
            value = el.get('src')
            if value is not None:
                value = try_urljoin(base_url, value)
        if value is not None:
            items = [' ', value, ' ']
    elif el.name == 'br':
        items = ['\n']
    else:
        for child in el.children:
            child_items = text_collection(child, replace_img, img_to_src, base_url)
            items.extend(child_items)
        if el.name == 'p':
            items = [P_BREAK_BEFORE] + items + [P_BREAK_AFTER, '\n']
    return items