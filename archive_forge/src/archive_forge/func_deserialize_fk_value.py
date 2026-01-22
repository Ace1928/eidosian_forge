from io import StringIO
from django.core.exceptions import ObjectDoesNotExist
from django.db import models
def deserialize_fk_value(field, field_value, using, handle_forward_references):
    if field_value is None:
        return None
    model = field.remote_field.model
    default_manager = model._default_manager
    field_name = field.remote_field.field_name
    if hasattr(default_manager, 'get_by_natural_key') and hasattr(field_value, '__iter__') and (not isinstance(field_value, str)):
        try:
            obj = default_manager.db_manager(using).get_by_natural_key(*field_value)
        except ObjectDoesNotExist:
            if handle_forward_references:
                return DEFER_FIELD
            else:
                raise
        value = getattr(obj, field_name)
        if model._meta.pk.remote_field:
            value = value.pk
        return value
    return model._meta.get_field(field_name).to_python(field_value)