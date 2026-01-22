from django.db.backends.postgresql.schema import DatabaseSchemaEditor
from django.db.models.expressions import Col, Func
def geo_quote_name(self, name):
    return self.connection.ops.geo_quote_name(name)