from decimal import Decimal
from django.conf import settings
from django.template import Library, Node, TemplateSyntaxError, Variable
from django.template.base import TokenType, render_value_in_context
from django.template.defaulttags import token_kwargs
from django.utils import translation
from django.utils.safestring import SafeData, SafeString, mark_safe
def render_token_list(self, tokens):
    result = []
    vars = []
    for token in tokens:
        if token.token_type == TokenType.TEXT:
            result.append(token.contents.replace('%', '%%'))
        elif token.token_type == TokenType.VAR:
            result.append('%%(%s)s' % token.contents)
            vars.append(token.contents)
    msg = ''.join(result)
    if self.trimmed:
        msg = translation.trim_whitespace(msg)
    return (msg, vars)