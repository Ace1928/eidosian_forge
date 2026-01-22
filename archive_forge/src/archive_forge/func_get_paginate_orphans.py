from django.core.exceptions import ImproperlyConfigured
from django.core.paginator import InvalidPage, Paginator
from django.db.models import QuerySet
from django.http import Http404
from django.utils.translation import gettext as _
from django.views.generic.base import ContextMixin, TemplateResponseMixin, View
def get_paginate_orphans(self):
    """
        Return the maximum number of orphans extend the last page by when
        paginating.
        """
    return self.paginate_orphans