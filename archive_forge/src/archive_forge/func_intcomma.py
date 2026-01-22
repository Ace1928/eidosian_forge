import re
from datetime import date, datetime, timezone
from decimal import Decimal
from django import template
from django.template import defaultfilters
from django.utils.formats import number_format
from django.utils.safestring import mark_safe
from django.utils.timezone import is_aware
from django.utils.translation import gettext as _
from django.utils.translation import (
@register.filter(is_safe=True)
def intcomma(value, use_l10n=True):
    """
    Convert an integer to a string containing commas every three digits.
    For example, 3000 becomes '3,000' and 45000 becomes '45,000'.
    """
    if use_l10n:
        try:
            if not isinstance(value, (float, Decimal)):
                value = int(value)
        except (TypeError, ValueError):
            return intcomma(value, False)
        else:
            return number_format(value, use_l10n=True, force_grouping=True)
    result = str(value)
    match = re.match('-?\\d+', result)
    if match:
        prefix = match[0]
        prefix_with_commas = re.sub('\\d{3}', '\\g<0>,', prefix[::-1])[::-1]
        prefix_with_commas = re.sub('^(-?),', '\\1', prefix_with_commas)
        result = prefix_with_commas + result[len(prefix):]
    return result