from decimal import Decimal
from django.conf import settings
from django.template import Library, Node, TemplateSyntaxError, Variable
from django.template.base import TokenType, render_value_in_context
from django.template.defaulttags import token_kwargs
from django.utils import translation
from django.utils.safestring import SafeData, SafeString, mark_safe
@register.filter
def language_name_translated(lang_code):
    english_name = translation.get_language_info(lang_code)['name']
    return translation.gettext(english_name)