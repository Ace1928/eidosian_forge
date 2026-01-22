from django.db import DatabaseError
from django.db.backends.sqlite3.schema import DatabaseSchemaEditor
def column_sql(self, model, field, include_default=False):
    from django.contrib.gis.db.models import GeometryField
    if not isinstance(field, GeometryField):
        return super().column_sql(model, field, include_default)
    self.geometry_sql.append(self.sql_add_geometry_column % {'table': self.geo_quote_name(model._meta.db_table), 'column': self.geo_quote_name(field.column), 'srid': field.srid, 'geom_type': self.geo_quote_name(field.geom_type), 'dim': field.dim, 'null': int(not field.null)})
    if field.spatial_index:
        self.geometry_sql.append(self.sql_add_spatial_index % {'table': self.quote_name(model._meta.db_table), 'column': self.quote_name(field.column)})
    return (None, None)