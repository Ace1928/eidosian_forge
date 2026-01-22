import functools
import sys
import threading
import warnings
from collections import Counter, defaultdict
from functools import partial
from django.core.exceptions import AppRegistryNotReady, ImproperlyConfigured
from .config import AppConfig
def check_models_ready(self):
    """Raise an exception if all models haven't been imported yet."""
    if not self.models_ready:
        raise AppRegistryNotReady("Models aren't loaded yet.")