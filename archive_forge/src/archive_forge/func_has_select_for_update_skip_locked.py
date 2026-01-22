import operator
from django.db.backends.base.features import BaseDatabaseFeatures
from django.utils.functional import cached_property
@cached_property
def has_select_for_update_skip_locked(self):
    if self.connection.mysql_is_mariadb:
        return self.connection.mysql_version >= (10, 6)
    return True