from io import StringIO
from django.core.exceptions import ObjectDoesNotExist
from django.db import models
def save_deferred_fields(self, using=None):
    self.m2m_data = {}
    for field, field_value in self.deferred_fields.items():
        opts = self.object._meta
        label = opts.app_label + '.' + opts.model_name
        if isinstance(field.remote_field, models.ManyToManyRel):
            try:
                values = deserialize_m2m_values(field, field_value, using, handle_forward_references=False)
            except M2MDeserializationError as e:
                raise DeserializationError.WithData(e.original_exc, label, self.object.pk, e.pk)
            self.m2m_data[field.name] = values
        elif isinstance(field.remote_field, models.ManyToOneRel):
            try:
                value = deserialize_fk_value(field, field_value, using, handle_forward_references=False)
            except Exception as e:
                raise DeserializationError.WithData(e, label, self.object.pk, field_value)
            setattr(self.object, field.attname, value)
    self.save()