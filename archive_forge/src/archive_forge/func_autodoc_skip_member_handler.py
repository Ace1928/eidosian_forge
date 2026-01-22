from inspect import signature
from docutils import nodes
from sphinx.domains.python import PyFunction
from sphinx.ext.autodoc import FunctionDocumenter
from celery.app.task import BaseTask
def autodoc_skip_member_handler(app, what, name, obj, skip, options):
    """Handler for autodoc-skip-member event."""
    if isinstance(obj, BaseTask) and getattr(obj, '__wrapped__'):
        if skip:
            return False
    return None