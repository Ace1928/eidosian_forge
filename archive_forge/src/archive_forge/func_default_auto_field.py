import inspect
import os
from importlib import import_module
from django.core.exceptions import ImproperlyConfigured
from django.utils.functional import cached_property
from django.utils.module_loading import import_string, module_has_submodule
@cached_property
def default_auto_field(self):
    from django.conf import settings
    return settings.DEFAULT_AUTO_FIELD