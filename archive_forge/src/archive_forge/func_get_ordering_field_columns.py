import warnings
from datetime import datetime, timedelta
from django import forms
from django.conf import settings
from django.contrib import messages
from django.contrib.admin import FieldListFilter
from django.contrib.admin.exceptions import (
from django.contrib.admin.options import (
from django.contrib.admin.utils import (
from django.core.exceptions import (
from django.core.paginator import InvalidPage
from django.db.models import F, Field, ManyToOneRel, OrderBy
from django.db.models.expressions import Combinable
from django.urls import reverse
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.http import urlencode
from django.utils.inspect import func_supports_parameter
from django.utils.timezone import make_aware
from django.utils.translation import gettext
def get_ordering_field_columns(self):
    """
        Return a dictionary of ordering field column numbers and asc/desc.
        """
    ordering = self._get_default_ordering()
    ordering_fields = {}
    if ORDER_VAR not in self.params:
        for field in ordering:
            if isinstance(field, (Combinable, OrderBy)):
                if not isinstance(field, OrderBy):
                    field = field.asc()
                if isinstance(field.expression, F):
                    order_type = 'desc' if field.descending else 'asc'
                    field = field.expression.name
                else:
                    continue
            elif field.startswith('-'):
                field = field.removeprefix('-')
                order_type = 'desc'
            else:
                order_type = 'asc'
            for index, attr in enumerate(self.list_display):
                if self.get_ordering_field(attr) == field:
                    ordering_fields[index] = order_type
                    break
    else:
        for p in self.params[ORDER_VAR].split('.'):
            none, pfx, idx = p.rpartition('-')
            try:
                idx = int(idx)
            except ValueError:
                continue
            ordering_fields[idx] = 'desc' if pfx == '-' else 'asc'
    return ordering_fields