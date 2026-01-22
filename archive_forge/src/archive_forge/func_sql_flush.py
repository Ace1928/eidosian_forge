import uuid
from django.conf import settings
from django.db.backends.base.operations import BaseDatabaseOperations
from django.db.backends.utils import split_tzname_delta
from django.db.models import Exists, ExpressionWrapper, Lookup
from django.db.models.constants import OnConflict
from django.utils import timezone
from django.utils.encoding import force_str
from django.utils.regex_helper import _lazy_re_compile
def sql_flush(self, style, tables, *, reset_sequences=False, allow_cascade=False):
    if not tables:
        return []
    sql = ['SET FOREIGN_KEY_CHECKS = 0;']
    if reset_sequences:
        sql.extend(('%s %s;' % (style.SQL_KEYWORD('TRUNCATE'), style.SQL_FIELD(self.quote_name(table_name))) for table_name in tables))
    else:
        sql.extend(('%s %s %s;' % (style.SQL_KEYWORD('DELETE'), style.SQL_KEYWORD('FROM'), style.SQL_FIELD(self.quote_name(table_name))) for table_name in tables))
    sql.append('SET FOREIGN_KEY_CHECKS = 1;')
    return sql