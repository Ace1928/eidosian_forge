import uuid
from django.conf import settings
from django.db.backends.base.operations import BaseDatabaseOperations
from django.db.backends.utils import split_tzname_delta
from django.db.models import Exists, ExpressionWrapper, Lookup
from django.db.models.constants import OnConflict
from django.utils import timezone
from django.utils.encoding import force_str
from django.utils.regex_helper import _lazy_re_compile
def conditional_expression_supported_in_where_clause(self, expression):
    if isinstance(expression, (Exists, Lookup)):
        return True
    if isinstance(expression, ExpressionWrapper) and expression.conditional:
        return self.conditional_expression_supported_in_where_clause(expression.expression)
    if getattr(expression, 'conditional', False):
        return False
    return super().conditional_expression_supported_in_where_clause(expression)