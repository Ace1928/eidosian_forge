from django.apps.registry import apps as global_apps
from django.db import migrations, router
from .exceptions import InvalidMigrationPlan
from .loader import MigrationLoader
from .recorder import MigrationRecorder
from .state import ProjectState
def check_replacements(self):
    """
        Mark replacement migrations applied if their replaced set all are.

        Do this unconditionally on every migrate, rather than just when
        migrations are applied or unapplied, to correctly handle the case
        when a new squash migration is pushed to a deployment that already had
        all its replaced migrations applied. In this case no new migration will
        be applied, but the applied state of the squashed migration must be
        maintained.
        """
    applied = self.recorder.applied_migrations()
    for key, migration in self.loader.replacements.items():
        all_applied = all((m in applied for m in migration.replaces))
        if all_applied and key not in applied:
            self.recorder.record_applied(*key)