import json
from collections import UserList
from django.conf import settings
from django.core.exceptions import ValidationError
from django.forms.renderers import get_default_renderer
from django.utils import timezone
from django.utils.html import escape, format_html_join
from django.utils.safestring import mark_safe
from django.utils.translation import gettext_lazy as _
def from_current_timezone(value):
    """
    When time zone support is enabled, convert naive datetimes
    entered in the current time zone to aware datetimes.
    """
    if settings.USE_TZ and value is not None and timezone.is_naive(value):
        current_timezone = timezone.get_current_timezone()
        try:
            if timezone._datetime_ambiguous_or_imaginary(value, current_timezone):
                raise ValueError('Ambiguous or non-existent time.')
            return timezone.make_aware(value, current_timezone)
        except Exception as exc:
            raise ValidationError(_('%(datetime)s couldn’t be interpreted in time zone %(current_timezone)s; it may be ambiguous or it may not exist.'), code='ambiguous_timezone', params={'datetime': value, 'current_timezone': current_timezone}) from exc
    return value