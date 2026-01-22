import operator
from django.db.backends.base.features import BaseDatabaseFeatures
from django.utils.functional import cached_property
@cached_property
def update_can_self_select(self):
    return self.connection.mysql_is_mariadb