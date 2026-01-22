import functools
import sys
import threading
import warnings
from collections import Counter, defaultdict
from functools import partial
from django.core.exceptions import AppRegistryNotReady, ImproperlyConfigured
from .config import AppConfig
def get_app_configs(self):
    """Import applications and return an iterable of app configs."""
    self.check_apps_ready()
    return self.app_configs.values()