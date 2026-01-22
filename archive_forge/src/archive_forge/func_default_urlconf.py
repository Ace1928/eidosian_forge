import functools
import inspect
import itertools
import re
import sys
import types
import warnings
from pathlib import Path
from django.conf import settings
from django.http import Http404, HttpResponse, HttpResponseNotFound
from django.template import Context, Engine, TemplateDoesNotExist
from django.template.defaultfilters import pprint
from django.urls import resolve
from django.utils import timezone
from django.utils.datastructures import MultiValueDict
from django.utils.encoding import force_str
from django.utils.module_loading import import_string
from django.utils.regex_helper import _lazy_re_compile
from django.utils.version import PY311, get_docs_version
from django.views.decorators.debug import coroutine_functions_to_sensitive_variables
def default_urlconf(request):
    """Create an empty URLconf 404 error response."""
    with builtin_template_path('default_urlconf.html').open(encoding='utf-8') as fh:
        t = DEBUG_ENGINE.from_string(fh.read())
    c = Context({'version': get_docs_version()})
    return HttpResponse(t.render(c))