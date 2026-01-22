import fnmatch
import os
import shutil
import subprocess
from pathlib import Path
from subprocess import run
from django.apps import apps as installed_apps
from django.utils.crypto import get_random_string
from django.utils.encoding import DEFAULT_LOCALE_ENCODING
from .base import CommandError, CommandParser
def parse_apps_and_model_labels(labels):
    """
    Parse a list of "app_label.ModelName" or "app_label" strings into actual
    objects and return a two-element tuple:
        (set of model classes, set of app_configs).
    Raise a CommandError if some specified models or apps don't exist.
    """
    apps = set()
    models = set()
    for label in labels:
        if '.' in label:
            try:
                model = installed_apps.get_model(label)
            except LookupError:
                raise CommandError('Unknown model: %s' % label)
            models.add(model)
        else:
            try:
                app_config = installed_apps.get_app_config(label)
            except LookupError as e:
                raise CommandError(str(e))
            apps.add(app_config)
    return (models, apps)