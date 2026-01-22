import functools
import os
from django.apps import apps
from django.conf import settings
from django.contrib.staticfiles import utils
from django.core.checks import Error, Warning
from django.core.exceptions import ImproperlyConfigured
from django.core.files.storage import FileSystemStorage, Storage, default_storage
from django.utils._os import safe_join
from django.utils.functional import LazyObject, empty
from django.utils.module_loading import import_string
def find_location(self, root, path, prefix=None):
    """
        Find a requested static file in a location and return the found
        absolute path (or ``None`` if no match).
        """
    if prefix:
        prefix = '%s%s' % (prefix, os.sep)
        if not path.startswith(prefix):
            return None
        path = path.removeprefix(prefix)
    path = safe_join(root, path)
    if os.path.exists(path):
        return path