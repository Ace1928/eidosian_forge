import functools
from django.db import connections
from django.db.backends.base.base import NO_DB_ALIAS
from django.db.backends.postgresql.psycopg_any import is_psycopg3
@functools.lru_cache
def get_citext_oids(connection_alias):
    """Return citext and citext array OIDs."""
    return get_type_oids(connection_alias, 'citext')