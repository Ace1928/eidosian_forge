import operator
from django.db.backends.base.features import BaseDatabaseFeatures
from django.utils.functional import cached_property
@cached_property
def supports_default_in_lead_lag(self):
    return not self.connection.mysql_is_mariadb