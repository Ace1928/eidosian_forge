from django.apps.registry import Apps
from django.db import DatabaseError, models
from django.utils.functional import classproperty
from django.utils.timezone import now
from .exceptions import MigrationSchemaMissing
def record_unapplied(self, app, name):
    """Record that a migration was unapplied."""
    self.ensure_schema()
    self.migration_qs.filter(app=app, name=name).delete()