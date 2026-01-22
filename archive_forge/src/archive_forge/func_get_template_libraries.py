import functools
from django.core.exceptions import ImproperlyConfigured
from django.utils.functional import cached_property
from django.utils.module_loading import import_string
from .base import Template
from .context import Context, _builtin_context_processors
from .exceptions import TemplateDoesNotExist
from .library import import_library
def get_template_libraries(self, libraries):
    loaded = {}
    for name, path in libraries.items():
        loaded[name] = import_library(path)
    return loaded