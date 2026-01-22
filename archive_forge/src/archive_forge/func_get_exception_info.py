from pathlib import Path
import jinja2
from django.conf import settings
from django.template import TemplateDoesNotExist, TemplateSyntaxError
from django.utils.functional import cached_property
from django.utils.module_loading import import_string
from .base import BaseEngine
from .utils import csrf_input_lazy, csrf_token_lazy
def get_exception_info(exception):
    """
    Format exception information for display on the debug page using the
    structure described in the template API documentation.
    """
    context_lines = 10
    lineno = exception.lineno
    source = exception.source
    if source is None:
        exception_file = Path(exception.filename)
        if exception_file.exists():
            source = exception_file.read_text()
    if source is not None:
        lines = list(enumerate(source.strip().split('\n'), start=1))
        during = lines[lineno - 1][1]
        total = len(lines)
        top = max(0, lineno - context_lines - 1)
        bottom = min(total, lineno + context_lines)
    else:
        during = ''
        lines = []
        total = top = bottom = 0
    return {'name': exception.filename, 'message': exception.message, 'source_lines': lines[top:bottom], 'line': lineno, 'before': '', 'during': during, 'after': '', 'total': total, 'top': top, 'bottom': bottom}