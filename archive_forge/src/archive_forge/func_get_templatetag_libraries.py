from importlib import import_module
from pkgutil import walk_packages
from django.apps import apps
from django.conf import settings
from django.template import TemplateDoesNotExist
from django.template.context import make_context
from django.template.engine import Engine
from django.template.library import InvalidTemplateLibrary
from .base import BaseEngine
def get_templatetag_libraries(self, custom_libraries):
    """
        Return a collation of template tag libraries from installed
        applications and the supplied custom_libraries argument.
        """
    libraries = get_installed_libraries()
    libraries.update(custom_libraries)
    return libraries