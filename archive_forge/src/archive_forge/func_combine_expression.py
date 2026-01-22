import uuid
from django.conf import settings
from django.db.backends.base.operations import BaseDatabaseOperations
from django.db.backends.utils import split_tzname_delta
from django.db.models import Exists, ExpressionWrapper, Lookup
from django.db.models.constants import OnConflict
from django.utils import timezone
from django.utils.encoding import force_str
from django.utils.regex_helper import _lazy_re_compile
def combine_expression(self, connector, sub_expressions):
    if connector == '^':
        return 'POW(%s)' % ','.join(sub_expressions)
    elif connector in ('&', '|', '<<', '#'):
        connector = '^' if connector == '#' else connector
        return 'CONVERT(%s, SIGNED)' % connector.join(sub_expressions)
    elif connector == '>>':
        lhs, rhs = sub_expressions
        return 'FLOOR(%(lhs)s / POW(2, %(rhs)s))' % {'lhs': lhs, 'rhs': rhs}
    return super().combine_expression(connector, sub_expressions)