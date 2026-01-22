from django.db.backends.oracle.introspection import DatabaseIntrospection
from django.db.backends.oracle.oracledb_any import oracledb
from django.utils.functional import cached_property
@cached_property
def data_types_reverse(self):
    return {**super().data_types_reverse, oracledb.DB_TYPE_OBJECT: 'GeometryField'}