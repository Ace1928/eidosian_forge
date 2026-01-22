import datetime
from django.contrib.admin.exceptions import NotRegistered
from django.contrib.admin.options import IncorrectLookupParameters
from django.contrib.admin.utils import (
from django.core.exceptions import ImproperlyConfigured, ValidationError
from django.db import models
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
def get_lookup_condition(self):
    lookup_conditions = []
    if self.field.empty_strings_allowed:
        lookup_conditions.append((self.field_path, ''))
    if self.field.null:
        lookup_conditions.append((f'{self.field_path}__isnull', True))
    return models.Q.create(lookup_conditions, connector=models.Q.OR)