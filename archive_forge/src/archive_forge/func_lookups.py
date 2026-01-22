import datetime
from django.contrib.admin.exceptions import NotRegistered
from django.contrib.admin.options import IncorrectLookupParameters
from django.contrib.admin.utils import (
from django.core.exceptions import ImproperlyConfigured, ValidationError
from django.db import models
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
def lookups(self, request, model_admin):
    """
        Must be overridden to return a list of tuples (value, verbose value)
        """
    raise NotImplementedError('The SimpleListFilter.lookups() method must be overridden to return a list of tuples (value, verbose value).')