import html
from html.parser import HTMLParser
from django.utils.html import VOID_ELEMENTS
from django.utils.regex_helper import _lazy_re_compile
def normalize_attributes(attributes):
    normalized = []
    for name, value in attributes:
        if name == 'class' and value:
            value = ' '.join(sorted((value for value in ASCII_WHITESPACE.split(value) if value)))
        if name in BOOLEAN_ATTRIBUTES:
            if not value or value == name:
                value = None
        elif value is None:
            value = ''
        normalized.append((name, value))
    return normalized