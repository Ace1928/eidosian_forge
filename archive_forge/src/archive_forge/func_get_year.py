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
def get_year(self):
    """Return the year for which this view should display data."""
    year = self.year
    if year is None:
        try:
            year = self.kwargs['year']
        except KeyError:
            try:
                year = self.request.GET['year']
            except KeyError:
                raise Http404(_('No year specified'))
    return year