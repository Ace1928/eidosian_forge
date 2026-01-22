import inspect
import logging
import re
from enum import Enum
from django.template.context import BaseContext
from django.utils.formats import localize
from django.utils.html import conditional_escape, escape
from django.utils.regex_helper import _lazy_re_compile
from django.utils.safestring import SafeData, SafeString, mark_safe
from django.utils.text import get_text_list, smart_split, unescape_string_literal
from django.utils.timezone import template_localtime
from django.utils.translation import gettext_lazy, pgettext_lazy
from .exceptions import TemplateSyntaxError
def split_contents(self):
    split = []
    bits = smart_split(self.contents)
    for bit in bits:
        if bit.startswith(('_("', "_('")):
            sentinel = bit[2] + ')'
            trans_bit = [bit]
            while not bit.endswith(sentinel):
                bit = next(bits)
                trans_bit.append(bit)
            bit = ' '.join(trans_bit)
        split.append(bit)
    return split