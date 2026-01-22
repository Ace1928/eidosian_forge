import operator
from django.db.backends.base.features import BaseDatabaseFeatures
from django.utils.functional import cached_property
@cached_property
def can_introspect_foreign_keys(self):
    """Confirm support for introspected foreign keys"""
    return self._mysql_storage_engine != 'MyISAM'