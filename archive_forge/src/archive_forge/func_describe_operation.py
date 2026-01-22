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
@staticmethod
def describe_operation(operation, backwards):
    """Return a string that describes a migration operation for --plan."""
    prefix = ''
    is_error = False
    if hasattr(operation, 'code'):
        code = operation.reverse_code if backwards else operation.code
        action = code.__doc__ or '' if code else None
    elif hasattr(operation, 'sql'):
        action = operation.reverse_sql if backwards else operation.sql
    else:
        action = ''
        if backwards:
            prefix = 'Undo '
    if action is not None:
        action = str(action).replace('\n', '')
    elif backwards:
        action = 'IRREVERSIBLE'
        is_error = True
    if action:
        action = ' -> ' + action
    truncated = Truncator(action)
    return (prefix + operation.describe() + truncated.chars(40), is_error)