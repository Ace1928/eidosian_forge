import functools
import sys
import threading
import warnings
from collections import Counter, defaultdict
from functools import partial
from django.core.exceptions import AppRegistryNotReady, ImproperlyConfigured
from .config import AppConfig
def do_pending_operations(self, model):
    """
        Take a newly-prepared model and pass it to each function waiting for
        it. This is called at the very end of Apps.register_model().
        """
    key = (model._meta.app_label, model._meta.model_name)
    for function in self._pending_operations.pop(key, []):
        function(model)