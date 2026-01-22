from django.core.exceptions import ImproperlyConfigured
from django.core.paginator import InvalidPage, Paginator
from django.db.models import QuerySet
from django.http import Http404
from django.utils.translation import gettext as _
from django.views.generic.base import ContextMixin, TemplateResponseMixin, View
def get_allow_empty(self):
    """
        Return ``True`` if the view should display empty lists and ``False``
        if a 404 should be raised instead.
        """
    return self.allow_empty