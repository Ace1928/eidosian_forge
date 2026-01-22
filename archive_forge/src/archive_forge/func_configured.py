import importlib
import os
import time
import traceback
import warnings
from pathlib import Path
import django
from django.conf import global_settings
from django.core.exceptions import ImproperlyConfigured
from django.utils.deprecation import RemovedInDjango51Warning, RemovedInDjango60Warning
from django.utils.functional import LazyObject, empty
@property
def configured(self):
    """Return True if the settings have already been configured."""
    return self._wrapped is not empty