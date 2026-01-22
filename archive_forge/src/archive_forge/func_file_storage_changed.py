import os
import time
import warnings
from asgiref.local import Local
from django.apps import apps
from django.core.exceptions import ImproperlyConfigured
from django.core.signals import setting_changed
from django.db import connections, router
from django.db.utils import ConnectionRouter
from django.dispatch import Signal, receiver
from django.utils import timezone
from django.utils.formats import FORMAT_SETTINGS, reset_format_cache
from django.utils.functional import empty
from django.utils.module_loading import import_string
@receiver(setting_changed)
def file_storage_changed(*, setting, **kwargs):
    if setting == 'DEFAULT_FILE_STORAGE':
        from django.conf import DEFAULT_STORAGE_ALIAS
        from django.core.files.storage import default_storage, storages
        try:
            del storages.backends
        except AttributeError:
            pass
        storages._storages[DEFAULT_STORAGE_ALIAS] = import_string(kwargs['value'])()
        default_storage._wrapped = empty