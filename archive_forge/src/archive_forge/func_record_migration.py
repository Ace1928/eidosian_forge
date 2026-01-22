from django.apps.registry import apps as global_apps
from django.db import migrations, router
from .exceptions import InvalidMigrationPlan
from .loader import MigrationLoader
from .recorder import MigrationRecorder
from .state import ProjectState
def record_migration(self, migration):
    if migration.replaces:
        for app_label, name in migration.replaces:
            self.recorder.record_applied(app_label, name)
    else:
        self.recorder.record_applied(migration.app_label, migration.name)