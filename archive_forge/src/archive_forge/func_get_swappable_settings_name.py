import functools
import sys
import threading
import warnings
from collections import Counter, defaultdict
from functools import partial
from django.core.exceptions import AppRegistryNotReady, ImproperlyConfigured
from .config import AppConfig
@functools.cache
def get_swappable_settings_name(self, to_string):
    """
        For a given model string (e.g. "auth.User"), return the name of the
        corresponding settings name if it refers to a swappable model. If the
        referred model is not swappable, return None.

        This method is decorated with @functools.cache because it's performance
        critical when it comes to migrations. Since the swappable settings don't
        change after Django has loaded the settings, there is no reason to get
        the respective settings attribute over and over again.
        """
    to_string = to_string.lower()
    for model in self.get_models(include_swapped=True):
        swapped = model._meta.swapped
        if swapped and swapped.lower() == to_string:
            return model._meta.swappable
        if model._meta.swappable and model._meta.label_lower == to_string:
            return model._meta.swappable
    return None