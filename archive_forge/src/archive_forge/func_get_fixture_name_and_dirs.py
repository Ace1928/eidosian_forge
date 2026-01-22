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
def get_fixture_name_and_dirs(self, fixture_name):
    dirname, basename = os.path.split(fixture_name)
    if os.path.isabs(fixture_name):
        fixture_dirs = [dirname]
    else:
        fixture_dirs = self.fixture_dirs
        if os.path.sep in os.path.normpath(fixture_name):
            fixture_dirs = [os.path.join(dir_, dirname) for dir_ in fixture_dirs]
    return (basename, fixture_dirs)