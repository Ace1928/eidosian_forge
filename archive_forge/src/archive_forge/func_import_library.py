from functools import wraps
from importlib import import_module
from inspect import getfullargspec, unwrap
from django.utils.html import conditional_escape
from django.utils.itercompat import is_iterable
from .base import Node, Template, token_kwargs
from .exceptions import TemplateSyntaxError
def import_library(name):
    """
    Load a Library object from a template tag module.
    """
    try:
        module = import_module(name)
    except ImportError as e:
        raise InvalidTemplateLibrary("Invalid template library specified. ImportError raised when trying to load '%s': %s" % (name, e))
    try:
        return module.register
    except AttributeError:
        raise InvalidTemplateLibrary("Module  %s does not have a variable named 'register'" % name)