from django.apps import apps as django_apps
from django.conf import settings
from django.core import paginator
from django.core.exceptions import ImproperlyConfigured
from django.utils import translation
def get_latest_lastmod(self):
    if self.date_field is not None:
        return self.queryset.order_by('-' + self.date_field).values_list(self.date_field, flat=True).first()
    return None