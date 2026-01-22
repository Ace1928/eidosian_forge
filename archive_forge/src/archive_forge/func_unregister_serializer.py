import os
import re
from importlib import import_module
from django import get_version
from django.apps import apps
from django.conf import SettingsReference  # NOQA
from django.db import migrations
from django.db.migrations.loader import MigrationLoader
from django.db.migrations.serializer import Serializer, serializer_factory
from django.utils.inspect import get_func_args
from django.utils.module_loading import module_dir
from django.utils.timezone import now
@classmethod
def unregister_serializer(cls, type_):
    Serializer.unregister(type_)