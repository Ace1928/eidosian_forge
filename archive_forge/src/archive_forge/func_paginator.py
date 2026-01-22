from django.apps import apps as django_apps
from django.conf import settings
from django.core import paginator
from django.core.exceptions import ImproperlyConfigured
from django.utils import translation
@property
def paginator(self):
    return paginator.Paginator(self._items(), self.limit)