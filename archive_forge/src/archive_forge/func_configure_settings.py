import pkgutil
from importlib import import_module
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.utils.connection import ConnectionDoesNotExist  # NOQA: F401
from django.utils.connection import BaseConnectionHandler
from django.utils.functional import cached_property
from django.utils.module_loading import import_string
def configure_settings(self, databases):
    databases = super().configure_settings(databases)
    if databases == {}:
        databases[DEFAULT_DB_ALIAS] = {'ENGINE': 'django.db.backends.dummy'}
    elif DEFAULT_DB_ALIAS not in databases:
        raise ImproperlyConfigured(f"You must define a '{DEFAULT_DB_ALIAS}' database.")
    elif databases[DEFAULT_DB_ALIAS] == {}:
        databases[DEFAULT_DB_ALIAS]['ENGINE'] = 'django.db.backends.dummy'
    for conn in databases.values():
        conn.setdefault('ATOMIC_REQUESTS', False)
        conn.setdefault('AUTOCOMMIT', True)
        conn.setdefault('ENGINE', 'django.db.backends.dummy')
        if conn['ENGINE'] == 'django.db.backends.' or not conn['ENGINE']:
            conn['ENGINE'] = 'django.db.backends.dummy'
        conn.setdefault('CONN_MAX_AGE', 0)
        conn.setdefault('CONN_HEALTH_CHECKS', False)
        conn.setdefault('OPTIONS', {})
        conn.setdefault('TIME_ZONE', None)
        for setting in ['NAME', 'USER', 'PASSWORD', 'HOST', 'PORT']:
            conn.setdefault(setting, '')
        test_settings = conn.setdefault('TEST', {})
        default_test_settings = [('CHARSET', None), ('COLLATION', None), ('MIGRATE', True), ('MIRROR', None), ('NAME', None)]
        for key, value in default_test_settings:
            test_settings.setdefault(key, value)
    return databases