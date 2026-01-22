import os
import platform as _platform
import re
from collections import namedtuple
from collections.abc import Mapping
from copy import deepcopy
from types import ModuleType
from kombu.utils.url import maybe_sanitize_url
from celery.exceptions import ImproperlyConfigured
from celery.platforms import pyimplementation
from celery.utils.collections import ConfigurationView
from celery.utils.imports import import_from_cwd, qualname, symbol_by_name
from celery.utils.text import pretty
from .defaults import _OLD_DEFAULTS, _OLD_SETTING_KEYS, _TO_NEW_KEY, _TO_OLD_KEY, DEFAULTS, SETTING_KEYS, find
def find_app(app, symbol_by_name=symbol_by_name, imp=import_from_cwd):
    """Find app by name."""
    from .base import Celery
    try:
        sym = symbol_by_name(app, imp=imp)
    except AttributeError:
        sym = imp(app)
    if isinstance(sym, ModuleType) and ':' not in app:
        try:
            found = sym.app
            if isinstance(found, ModuleType):
                raise AttributeError()
        except AttributeError:
            try:
                found = sym.celery
                if isinstance(found, ModuleType):
                    raise AttributeError("attribute 'celery' is the celery module not the instance of celery")
            except AttributeError:
                if getattr(sym, '__path__', None):
                    try:
                        return find_app(f'{app}.celery', symbol_by_name=symbol_by_name, imp=imp)
                    except ImportError:
                        pass
                for suspect in vars(sym).values():
                    if isinstance(suspect, Celery):
                        return suspect
                raise
            else:
                return found
        else:
            return found
    return sym