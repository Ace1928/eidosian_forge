from django.template import Library, Node, TemplateSyntaxError
from django.utils import formats
@register.tag('localize')
def localize_tag(parser, token):
    """
    Force or prevents localization of values.

    Sample usage::

        {% localize off %}
            var pi = {{ 3.1415 }};
        {% endlocalize %}
    """
    use_l10n = None
    bits = list(token.split_contents())
    if len(bits) == 1:
        use_l10n = True
    elif len(bits) > 2 or bits[1] not in ('on', 'off'):
        raise TemplateSyntaxError("%r argument should be 'on' or 'off'" % bits[0])
    else:
        use_l10n = bits[1] == 'on'
    nodelist = parser.parse(('endlocalize',))
    parser.delete_first_token()
    return LocalizeNode(nodelist, use_l10n)