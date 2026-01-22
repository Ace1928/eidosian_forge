from django.apps import apps as django_apps
from django.conf import settings
from django.core import paginator
from django.core.exceptions import ImproperlyConfigured
from django.utils import translation
def lastmod(self, item):
    if self.date_field is not None:
        return getattr(item, self.date_field)
    return None