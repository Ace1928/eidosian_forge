import mimetypes
import posixpath
from pathlib import Path
from django.http import FileResponse, Http404, HttpResponse, HttpResponseNotModified
from django.template import Context, Engine, TemplateDoesNotExist, loader
from django.utils._os import safe_join
from django.utils.http import http_date, parse_http_date
from django.utils.translation import gettext as _
from django.utils.translation import gettext_lazy
def builtin_template_path(name):
    """
    Return a path to a builtin template.

    Avoid calling this function at the module level or in a class-definition
    because __file__ may not exist, e.g. in frozen environments.
    """
    return Path(__file__).parent / 'templates' / name