import re
import sys
import warnings
from collections import namedtuple
from datetime import datetime
from itertools import cycle as itertools_cycle
from itertools import groupby
from django.conf import settings
from django.utils import timezone
from django.utils.html import conditional_escape, escape, format_html
from django.utils.lorem_ipsum import paragraphs, words
from django.utils.safestring import mark_safe
from .base import (
from .context import Context
from .defaultfilters import date
from .library import Library
from .smartif import IfParser, Literal
@register.tag('filter')
def do_filter(parser, token):
    """
    Filter the contents of the block through variable filters.

    Filters can also be piped through each other, and they can have
    arguments -- just like in variable syntax.

    Sample usage::

        {% filter force_escape|lower %}
            This text will be HTML-escaped, and will appear in lowercase.
        {% endfilter %}

    Note that the ``escape`` and ``safe`` filters are not acceptable arguments.
    Instead, use the ``autoescape`` tag to manage autoescaping for blocks of
    template code.
    """
    _, rest = token.contents.split(None, 1)
    filter_expr = parser.compile_filter('var|%s' % rest)
    for func, unused in filter_expr.filters:
        filter_name = getattr(func, '_filter_name', None)
        if filter_name in ('escape', 'safe'):
            raise TemplateSyntaxError('"filter %s" is not permitted.  Use the "autoescape" tag instead.' % filter_name)
    nodelist = parser.parse(('endfilter',))
    parser.delete_first_token()
    return FilterNode(filter_expr, nodelist)