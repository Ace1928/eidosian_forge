import copy
import enum
import json
import re
from functools import partial, update_wrapper
from urllib.parse import parse_qsl
from urllib.parse import quote as urlquote
from urllib.parse import urlparse
from django import forms
from django.conf import settings
from django.contrib import messages
from django.contrib.admin import helpers, widgets
from django.contrib.admin.checks import (
from django.contrib.admin.exceptions import DisallowedModelAdminToField, NotRegistered
from django.contrib.admin.templatetags.admin_urls import add_preserved_filters
from django.contrib.admin.utils import (
from django.contrib.admin.widgets import AutocompleteSelect, AutocompleteSelectMultiple
from django.contrib.auth import get_permission_codename
from django.core.exceptions import (
from django.core.paginator import Paginator
from django.db import models, router, transaction
from django.db.models.constants import LOOKUP_SEP
from django.forms.formsets import DELETION_FIELD_NAME, all_valid
from django.forms.models import (
from django.forms.widgets import CheckboxSelectMultiple, SelectMultiple
from django.http import HttpResponseRedirect
from django.http.response import HttpResponseBase
from django.template.response import SimpleTemplateResponse, TemplateResponse
from django.urls import reverse
from django.utils.decorators import method_decorator
from django.utils.html import format_html
from django.utils.http import urlencode
from django.utils.safestring import mark_safe
from django.utils.text import (
from django.utils.translation import gettext as _
from django.utils.translation import ngettext
from django.views.decorators.csrf import csrf_protect
from django.views.generic import RedirectView
def get_search_results(self, request, queryset, search_term):
    """
        Return a tuple containing a queryset to implement the search
        and a boolean indicating if the results may contain duplicates.
        """

    def construct_search(field_name):
        if field_name.startswith('^'):
            return '%s__istartswith' % field_name.removeprefix('^')
        elif field_name.startswith('='):
            return '%s__iexact' % field_name.removeprefix('=')
        elif field_name.startswith('@'):
            return '%s__search' % field_name.removeprefix('@')
        opts = queryset.model._meta
        lookup_fields = field_name.split(LOOKUP_SEP)
        prev_field = None
        for path_part in lookup_fields:
            if path_part == 'pk':
                path_part = opts.pk.name
            try:
                field = opts.get_field(path_part)
            except FieldDoesNotExist:
                if prev_field and prev_field.get_lookup(path_part):
                    return field_name
            else:
                prev_field = field
                if hasattr(field, 'path_infos'):
                    opts = field.path_infos[-1].to_opts
        return '%s__icontains' % field_name
    may_have_duplicates = False
    search_fields = self.get_search_fields(request)
    if search_fields and search_term:
        orm_lookups = [construct_search(str(search_field)) for search_field in search_fields]
        term_queries = []
        for bit in smart_split(search_term):
            if bit.startswith(('"', "'")) and bit[0] == bit[-1]:
                bit = unescape_string_literal(bit)
            or_queries = models.Q.create([(orm_lookup, bit) for orm_lookup in orm_lookups], connector=models.Q.OR)
            term_queries.append(or_queries)
        queryset = queryset.filter(models.Q.create(term_queries))
        may_have_duplicates |= any((lookup_spawns_duplicates(self.opts, search_spec) for search_spec in orm_lookups))
    return (queryset, may_have_duplicates)