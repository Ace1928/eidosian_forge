from functools import lru_cache
from django.db.backends.base.base import NO_DB_ALIAS
from django.db.backends.postgresql.base import DatabaseWrapper as PsycopgDatabaseWrapper
from django.db.backends.postgresql.features import (
from django.db.backends.postgresql.introspection import (
from django.db.backends.postgresql.operations import (
from django.db.backends.postgresql.psycopg_any import is_psycopg3
from .adapter import PostGISAdapter
from .features import DatabaseFeatures
from .introspection import PostGISIntrospection
from .operations import PostGISOperations
from .schema import PostGISSchemaEditor
def register_geometry_adapters(self, pg_connection, clear_caches=False):
    if clear_caches:
        for typename in self._type_infos:
            self._type_infos[typename].pop(self.alias, None)
    geo_oid = self._register_type(pg_connection, 'geometry')
    geog_oid = self._register_type(pg_connection, 'geography')
    raster_oid = self._register_type(pg_connection, 'raster')
    PostGISTextDumper, PostGISBinaryDumper = postgis_adapters(geo_oid, geog_oid, raster_oid)
    pg_connection.adapters.register_dumper(PostGISAdapter, PostGISTextDumper)
    pg_connection.adapters.register_dumper(PostGISAdapter, PostGISBinaryDumper)