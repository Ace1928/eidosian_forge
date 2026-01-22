import os
import shutil
from django.apps import apps
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.core.management.utils import run_formatters
from django.db import DEFAULT_DB_ALIAS, connections, migrations
from django.db.migrations.loader import AmbiguityError, MigrationLoader
from django.db.migrations.migration import SwappableTuple
from django.db.migrations.optimizer import MigrationOptimizer
from django.db.migrations.writer import MigrationWriter
from django.utils.version import get_docs_version
def find_migration(self, loader, app_label, name):
    try:
        return loader.get_migration_by_prefix(app_label, name)
    except AmbiguityError:
        raise CommandError("More than one migration matches '%s' in app '%s'. Please be more specific." % (name, app_label))
    except KeyError:
        raise CommandError("Cannot find a migration matching '%s' from app '%s'." % (name, app_label))