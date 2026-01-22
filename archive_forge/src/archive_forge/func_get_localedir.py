import gettext
import os
import re
import textwrap
import warnings
from . import declarative
def get_localedir():
    """Retrieve the location of locales."""
    file_dir = os.path.join(os.path.dirname(__file__), 'i18n')
    if not hasattr(os, 'access'):
        return file_dir
    import importlib.resources as resources
    pkg_name = __name__.split('.', 1)[0]
    try:
        resource_dir = resources.files(pkg_name) / 'i18n'
    except AttributeError:
        with resources.path(pkg_name, 'i18n') as resource_dir:
            pass
    if os.access(resource_dir, os.R_OK | os.X_OK):
        return resource_dir
    if os.access(file_dir, os.R_OK | os.X_OK):
        return file_dir
    return os.path.normpath('/usr/share/locale')