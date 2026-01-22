from decimal import Decimal
from django.conf import settings
from django.template import Library, Node, TemplateSyntaxError, Variable
from django.template.base import TokenType, render_value_in_context
from django.template.defaulttags import token_kwargs
from django.utils import translation
from django.utils.safestring import SafeData, SafeString, mark_safe
@register.tag('get_current_language_bidi')
def do_get_current_language_bidi(parser, token):
    """
    Store the current language layout in the context.

    Usage::

        {% get_current_language_bidi as bidi %}

    This fetches the currently active language's layout and puts its value into
    the ``bidi`` context variable. True indicates right-to-left layout,
    otherwise left-to-right.
    """
    args = token.contents.split()
    if len(args) != 3 or args[1] != 'as':
        raise TemplateSyntaxError("'get_current_language_bidi' requires 'as variable' (got %r)" % args)
    return GetCurrentLanguageBidiNode(args[2])