import functools
from django.core.exceptions import ImproperlyConfigured
from django.utils.functional import cached_property
from django.utils.module_loading import import_string
from .base import Template
from .context import Context, _builtin_context_processors
from .exceptions import TemplateDoesNotExist
from .library import import_library
def render_to_string(self, template_name, context=None):
    """
        Render the template specified by template_name with the given context.
        For use in Django's test suite.
        """
    if isinstance(template_name, (list, tuple)):
        t = self.select_template(template_name)
    else:
        t = self.get_template(template_name)
    if isinstance(context, Context):
        return t.render(context)
    else:
        return t.render(Context(context, autoescape=self.autoescape))