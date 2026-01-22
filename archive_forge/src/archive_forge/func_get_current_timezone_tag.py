import zoneinfo
from datetime import datetime
from datetime import timezone as datetime_timezone
from datetime import tzinfo
from django.template import Library, Node, TemplateSyntaxError
from django.utils import timezone
@register.tag('get_current_timezone')
def get_current_timezone_tag(parser, token):
    """
    Store the name of the current time zone in the context.

    Usage::

        {% get_current_timezone as TIME_ZONE %}

    This will fetch the currently active time zone and put its name
    into the ``TIME_ZONE`` context variable.
    """
    args = token.contents.split()
    if len(args) != 3 or args[1] != 'as':
        raise TemplateSyntaxError("'get_current_timezone' requires 'as variable' (got %r)" % args)
    return GetCurrentTimezoneNode(args[2])