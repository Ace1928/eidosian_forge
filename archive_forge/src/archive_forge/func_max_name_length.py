from django.db import NotSupportedError
from django.db.models import Func, Index
from django.utils.functional import cached_property
@cached_property
def max_name_length(self):
    return Index.max_name_length - len(Index.suffix) + len(self.suffix)