import uuid
from django.conf import settings
from django.db.backends.base.operations import BaseDatabaseOperations
from django.db.backends.utils import split_tzname_delta
from django.db.models import Exists, ExpressionWrapper, Lookup
from django.db.models.constants import OnConflict
from django.utils import timezone
from django.utils.encoding import force_str
from django.utils.regex_helper import _lazy_re_compile
def regex_lookup(self, lookup_type):
    if self.connection.mysql_is_mariadb:
        if lookup_type == 'regex':
            return '%s REGEXP BINARY %s'
        return '%s REGEXP %s'
    match_option = 'c' if lookup_type == 'regex' else 'i'
    return "REGEXP_LIKE(%%s, %%s, '%s')" % match_option