import datetime
from django.contrib.admin.templatetags.admin_urls import add_preserved_filters
from django.contrib.admin.utils import (
from django.contrib.admin.views.main import (
from django.core.exceptions import ObjectDoesNotExist
from django.db import models
from django.template import Library
from django.template.loader import get_template
from django.templatetags.static import static
from django.urls import NoReverseMatch
from django.utils import formats, timezone
from django.utils.html import format_html
from django.utils.safestring import mark_safe
from django.utils.text import capfirst
from django.utils.translation import gettext as _
from .base import InclusionAdminNode
def search_form(cl):
    """
    Display a search form for searching the list.
    """
    return {'cl': cl, 'show_result_count': cl.result_count != cl.full_result_count, 'search_var': SEARCH_VAR, 'is_popup_var': IS_POPUP_VAR, 'is_facets_var': IS_FACETS_VAR}