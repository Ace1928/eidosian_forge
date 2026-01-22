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
def user_model_swapped(*, setting, **kwargs):
    if setting == 'AUTH_USER_MODEL':
        apps.clear_cache()
        try:
            from django.contrib.auth import get_user_model
            UserModel = get_user_model()
        except ImproperlyConfigured:
            pass
        else:
            from django.contrib.auth import backends
            backends.UserModel = UserModel
            from django.contrib.auth import forms
            forms.UserModel = UserModel
            from django.contrib.auth.handlers import modwsgi
            modwsgi.UserModel = UserModel
            from django.contrib.auth.management.commands import changepassword
            changepassword.UserModel = UserModel
            from django.contrib.auth import views
            views.UserModel = UserModel