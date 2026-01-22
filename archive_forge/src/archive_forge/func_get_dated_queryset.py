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
def get_dated_queryset(self, **lookup):
    """
        Get a queryset properly filtered according to `allow_future` and any
        extra lookup kwargs.
        """
    qs = self.get_queryset().filter(**lookup)
    date_field = self.get_date_field()
    allow_future = self.get_allow_future()
    allow_empty = self.get_allow_empty()
    paginate_by = self.get_paginate_by(qs)
    if not allow_future:
        now = timezone.now() if self.uses_datetime_field else timezone_today()
        qs = qs.filter(**{'%s__lte' % date_field: now})
    if not allow_empty:
        is_empty = not qs if paginate_by is None else not qs.exists()
        if is_empty:
            raise Http404(_('No %(verbose_name_plural)s available') % {'verbose_name_plural': qs.model._meta.verbose_name_plural})
    return qs