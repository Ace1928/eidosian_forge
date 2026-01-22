import functools
import glob
import gzip
import os
import sys
import warnings
import zipfile
from itertools import product
from django.apps import apps
from django.conf import settings
from django.core import serializers
from django.core.exceptions import ImproperlyConfigured
from django.core.management.base import BaseCommand, CommandError
from django.core.management.color import no_style
from django.core.management.utils import parse_apps_and_model_labels
from django.db import (
from django.utils.functional import cached_property
@cached_property
def fixture_dirs(self):
    """
        Return a list of fixture directories.

        The list contains the 'fixtures' subdirectory of each installed
        application, if it exists, the directories in FIXTURE_DIRS, and the
        current directory.
        """
    dirs = []
    fixture_dirs = settings.FIXTURE_DIRS
    if len(fixture_dirs) != len(set(fixture_dirs)):
        raise ImproperlyConfigured('settings.FIXTURE_DIRS contains duplicates.')
    for app_config in apps.get_app_configs():
        app_label = app_config.label
        app_dir = os.path.join(app_config.path, 'fixtures')
        if app_dir in [str(d) for d in fixture_dirs]:
            raise ImproperlyConfigured("'%s' is a default fixture directory for the '%s' app and cannot be listed in settings.FIXTURE_DIRS." % (app_dir, app_label))
        if self.app_label and app_label != self.app_label:
            continue
        if os.path.isdir(app_dir):
            dirs.append(app_dir)
    dirs.extend(fixture_dirs)
    dirs.append('')
    return [os.path.realpath(d) for d in dirs]