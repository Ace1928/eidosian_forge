import operator
from django.db.backends.base.features import BaseDatabaseFeatures
from django.utils.functional import cached_property
@cached_property
def can_return_columns_from_insert(self):
    return self.connection.mysql_is_mariadb and self.connection.mysql_version >= (10, 5, 0)