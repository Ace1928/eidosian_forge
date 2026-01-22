from django.db import DatabaseError
from django.db.backends.sqlite3.schema import DatabaseSchemaEditor
def remove_field(self, model, field):
    from django.contrib.gis.db.models import GeometryField
    if isinstance(field, GeometryField):
        self._remake_table(model, delete_field=field)
    else:
        super().remove_field(model, field)