import sys
from django.apps import apps
from django.core.management.base import BaseCommand
from django.db import DEFAULT_DB_ALIAS, connections
from django.db.migrations.loader import MigrationLoader
from django.db.migrations.recorder import MigrationRecorder
def print_deps(node):
    out = []
    for parent in sorted(node.parents):
        out.append('%s.%s' % parent.key)
    if out:
        return ' ... (%s)' % ', '.join(out)
    return ''