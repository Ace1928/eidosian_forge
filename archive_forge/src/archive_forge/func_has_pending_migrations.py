import importlib
import os.path
import pkgutil
from glance.common import exception
from glance.db import migration as db_migrations
from glance.db.sqlalchemy import api as db_api
def has_pending_migrations(engine=None, release=db_migrations.CURRENT_RELEASE):
    if not engine:
        engine = db_api.get_engine()
    migrations = _find_migration_modules(release)
    if not migrations:
        return False
    return any([x.has_migrations(engine) for x in migrations])