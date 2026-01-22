from django.db.migrations.utils import field_references
from django.db.models import NOT_PROVIDED
from django.utils.functional import cached_property
from .base import Operation
def references_model(self, name, app_label):
    name_lower = name.lower()
    if name_lower == self.model_name_lower:
        return True
    if self.field:
        return bool(field_references((app_label, self.model_name_lower), self.field, (app_label, name_lower)))
    return False