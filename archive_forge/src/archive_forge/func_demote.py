import warnings
from django.core.exceptions import FullResultSet
from django.db.models.sql.constants import INNER, LOUTER
from django.utils.deprecation import RemovedInDjango60Warning
def demote(self):
    new = self.relabeled_clone({})
    new.join_type = INNER
    return new