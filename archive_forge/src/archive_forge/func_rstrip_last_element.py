import html
from html.parser import HTMLParser
from django.utils.html import VOID_ELEMENTS
from django.utils.regex_helper import _lazy_re_compile
def rstrip_last_element(children):
    if children and isinstance(children[-1], str):
        children[-1] = children[-1].rstrip()
        if not children[-1]:
            children.pop()
            children = rstrip_last_element(children)
    return children