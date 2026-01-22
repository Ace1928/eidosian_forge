from decimal import Decimal
from django.conf import settings
from django.template import Library, Node, TemplateSyntaxError, Variable
from django.template.base import TokenType, render_value_in_context
from django.template.defaulttags import token_kwargs
from django.utils import translation
from django.utils.safestring import SafeData, SafeString, mark_safe
def render_value(key):
    if key in context:
        val = context[key]
    else:
        val = default_value % key if '%s' in default_value else default_value
    return render_value_in_context(val, context)