from django.db import NotSupportedError
from django.db.models.expressions import Func, Value
from django.db.models.fields import TextField
from django.db.models.fields.json import JSONField
from django.utils.regex_helper import _lazy_re_compile
@property
def empty_result_set_value(self):
    for expression in self.get_source_expressions():
        result = expression.empty_result_set_value
        if result is NotImplemented or result is not None:
            return result
    return None