from django.db.migrations.utils import field_references
from django.db.models import NOT_PROVIDED
from django.utils.functional import cached_property
from .base import Operation
def is_same_field_operation(self, operation):
    return self.is_same_model_operation(operation) and self.name_lower == operation.name_lower