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
def find_in_app(self, app, path):
    """
        Find a requested static file in an app's static locations.
        """
    storage = self.storages.get(app)
    if storage and storage.exists(path):
        matched_path = storage.path(path)
        if matched_path:
            return matched_path