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
def get_queryset(self, request, exclude_parameters=None):
    self.filter_specs, self.has_filters, remaining_lookup_params, filters_may_have_duplicates, self.has_active_filters = self.get_filters(request)
    qs = self.root_queryset
    for filter_spec in self.filter_specs:
        if exclude_parameters is None or filter_spec.expected_parameters() != exclude_parameters:
            new_qs = filter_spec.queryset(request, qs)
            if new_qs is not None:
                qs = new_qs
    try:
        q_object = build_q_object_from_lookup_parameters(remaining_lookup_params)
        qs = qs.filter(q_object)
    except (SuspiciousOperation, ImproperlyConfigured):
        raise
    except Exception as e:
        raise IncorrectLookupParameters(e)
    if not qs.query.select_related:
        qs = self.apply_select_related(qs)
    ordering = self.get_ordering(request, qs)
    qs = qs.order_by(*ordering)
    qs, search_may_have_duplicates = self.model_admin.get_search_results(request, qs, self.query)
    self.clear_all_filters_qs = self.get_query_string(new_params=remaining_lookup_params, remove=self.get_filters_params())
    if filters_may_have_duplicates | search_may_have_duplicates:
        return qs.distinct()
    else:
        return qs