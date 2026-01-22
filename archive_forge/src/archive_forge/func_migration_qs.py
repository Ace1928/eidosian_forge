from django.apps.registry import Apps
from django.db import DatabaseError, models
from django.utils.functional import classproperty
from django.utils.timezone import now
from .exceptions import MigrationSchemaMissing
@property
def migration_qs(self):
    return self.Migration.objects.using(self.connection.alias)