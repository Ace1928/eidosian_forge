import copy
import inspect
import warnings
from functools import partialmethod
from itertools import chain
from asgiref.sync import sync_to_async
import django
from django.apps import apps
from django.conf import settings
from django.core import checks
from django.core.exceptions import (
from django.db import (
from django.db.models import NOT_PROVIDED, ExpressionWrapper, IntegerField, Max, Value
from django.db.models.constants import LOOKUP_SEP
from django.db.models.constraints import CheckConstraint, UniqueConstraint
from django.db.models.deletion import CASCADE, Collector
from django.db.models.expressions import DatabaseDefault, RawSQL
from django.db.models.fields.related import (
from django.db.models.functions import Coalesce
from django.db.models.manager import Manager
from django.db.models.options import Options
from django.db.models.query import F, Q
from django.db.models.signals import (
from django.db.models.utils import AltersData, make_model_tuple
from django.utils.encoding import force_str
from django.utils.hashable import make_hashable
from django.utils.text import capfirst, get_text_list
from django.utils.translation import gettext_lazy as _
def clean_fields(self, exclude=None):
    """
        Clean all fields and raise a ValidationError containing a dict
        of all validation errors if any occur.
        """
    if exclude is None:
        exclude = set()
    errors = {}
    for f in self._meta.fields:
        if f.name in exclude or f.generated:
            continue
        raw_value = getattr(self, f.attname)
        if f.blank and raw_value in f.empty_values:
            continue
        if isinstance(raw_value, DatabaseDefault):
            continue
        try:
            setattr(self, f.attname, f.clean(raw_value, self))
        except ValidationError as e:
            errors[f.name] = e.error_list
    if errors:
        raise ValidationError(errors)