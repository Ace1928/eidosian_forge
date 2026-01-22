from django.apps.registry import apps as global_apps
from django.db import migrations, router
from .exceptions import InvalidMigrationPlan
from .loader import MigrationLoader
from .recorder import MigrationRecorder
from .state import ProjectState
def should_skip_detecting_model(migration, model):
    """
            No need to detect tables for proxy models, unmanaged models, or
            models that can't be migrated on the current database.
            """
    return model._meta.proxy or not model._meta.managed or (not router.allow_migrate(self.connection.alias, migration.app_label, model_name=model._meta.model_name))