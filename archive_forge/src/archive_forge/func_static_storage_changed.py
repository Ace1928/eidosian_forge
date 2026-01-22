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
def static_storage_changed(*, setting, **kwargs):
    if setting in {'STATICFILES_STORAGE', 'STATIC_ROOT', 'STATIC_URL'}:
        from django.contrib.staticfiles.storage import staticfiles_storage
        staticfiles_storage._wrapped = empty
    if setting == 'STATICFILES_STORAGE':
        from django.conf import STATICFILES_STORAGE_ALIAS
        from django.core.files.storage import storages
        try:
            del storages.backends
        except AttributeError:
            pass
        storages._storages[STATICFILES_STORAGE_ALIAS] = import_string(kwargs['value'])()