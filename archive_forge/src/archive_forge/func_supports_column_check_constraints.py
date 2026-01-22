import operator
from django.db.backends.base.features import BaseDatabaseFeatures
from django.utils.functional import cached_property
@cached_property
def supports_column_check_constraints(self):
    if self.connection.mysql_is_mariadb:
        return True
    return self.connection.mysql_version >= (8, 0, 16)