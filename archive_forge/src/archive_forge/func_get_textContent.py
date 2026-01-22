import re
from urllib.parse import urljoin
import bs4
from bs4.element import Comment, NavigableString, Tag
def get_textContent(el, replace_img=False, img_to_src=True, base_url=''):
    """Get the text content of an element, replacing images by alt or src"""
    DROP_TAGS = ('script', 'style', 'template')
    PRE_TAGS = ('pre',)
    P_BREAK_BEFORE = 1
    P_BREAK_AFTER = 0
    PRE_BEFORE = 2
    PRE_AFTER = 3

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
    results = [t for t in text_collection(el, replace_img, img_to_src, base_url) if t != '']
    if results:
        length = len(results)
        for i in range(0, length):
            if results[i] == ' ' and (i == 0 or i == length - 1 or results[i - 1] == ' ' or (results[i - 1] in (P_BREAK_BEFORE, P_BREAK_AFTER)) or (results[i + 1] == ' ') or (results[i + 1] in (P_BREAK_BEFORE, P_BREAK_AFTER))):
                results[i] = ''
    if results:
        while isinstance(results[0], str) and (results[0] == '' or results[0].isspace()) or results[0] in (P_BREAK_BEFORE, P_BREAK_AFTER):
            results.pop(0)
            if not results:
                break
    if results:
        while isinstance(results[-1], str) and (results[-1] == '' or results[-1].isspace()) or results[-1] in (P_BREAK_BEFORE, P_BREAK_AFTER):
            results.pop(-1)
            if not results:
                break
    if results:
        if isinstance(results[0], str):
            results[0] = results[0].lstrip()
        if isinstance(results[-1], str):
            results[-1] = results[-1].rstrip()
    text = ''
    count = 0
    last = None
    for t in results:
        if t in (P_BREAK_BEFORE, P_BREAK_AFTER):
            count = max(t, count)
        elif t == PRE_BEFORE:
            text = text.rstrip(' ')
        elif not isinstance(t, int):
            if count or last == '\n':
                t = t.lstrip(' ')
            text = ''.join([text, '\n' * count, t])
            count = 0
        last = t
    return text