from django.db import DatabaseError, InterfaceError
from django.db.backends.base.features import BaseDatabaseFeatures
from django.utils.functional import cached_property
@cached_property
def supports_primitives_in_json_field(self):
    return self.connection.oracle_version >= (21,)