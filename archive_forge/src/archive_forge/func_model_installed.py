import sys
import time
from importlib import import_module
from django.apps import apps
from django.core.management.base import BaseCommand, CommandError, no_translations
from django.core.management.sql import emit_post_migrate_signal, emit_pre_migrate_signal
from django.db import DEFAULT_DB_ALIAS, connections, router
from django.db.migrations.autodetector import MigrationAutodetector
from django.db.migrations.executor import MigrationExecutor
from django.db.migrations.loader import AmbiguityError
from django.db.migrations.state import ModelState, ProjectState
from django.utils.module_loading import module_has_submodule
from django.utils.text import Truncator
def model_installed(model):
    opts = model._meta
    converter = connection.introspection.identifier_converter
    return not (converter(opts.db_table) in tables or (opts.auto_created and converter(opts.auto_created._meta.db_table) in tables))