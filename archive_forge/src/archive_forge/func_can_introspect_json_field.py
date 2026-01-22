import operator
from django.db.backends.base.features import BaseDatabaseFeatures
from django.utils.functional import cached_property
@cached_property
def can_introspect_json_field(self):
    if self.connection.mysql_is_mariadb:
        return self.can_introspect_check_constraints
    return True