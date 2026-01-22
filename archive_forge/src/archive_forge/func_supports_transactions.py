import operator
from django.db.backends.base.features import BaseDatabaseFeatures
from django.utils.functional import cached_property
@cached_property
def supports_transactions(self):
    """
        All storage engines except MyISAM support transactions.
        """
    return self._mysql_storage_engine != 'MyISAM'