import datetime
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.db import models
from django.http import Http404
from django.utils import timezone
from django.utils.functional import cached_property
from django.utils.translation import gettext as _
from django.views.generic.base import View
from django.views.generic.detail import (
from django.views.generic.list import (
def get_date_list(self, queryset, date_type=None, ordering='ASC'):
    """
        Get a date list by calling `queryset.dates/datetimes()`, checking
        along the way for empty lists that aren't allowed.
        """
    date_field = self.get_date_field()
    allow_empty = self.get_allow_empty()
    if date_type is None:
        date_type = self.get_date_list_period()
    if self.uses_datetime_field:
        date_list = queryset.datetimes(date_field, date_type, ordering)
    else:
        date_list = queryset.dates(date_field, date_type, ordering)
    if date_list is not None and (not date_list) and (not allow_empty):
        raise Http404(_('No %(verbose_name_plural)s available') % {'verbose_name_plural': queryset.model._meta.verbose_name_plural})
    return date_list