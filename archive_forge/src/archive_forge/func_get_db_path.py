import datetime
import os
from oslo_concurrency import lockutils
from oslo_config import cfg
from oslo_log import log as logging
from glance.common import exception
from glance import context
import glance.db
from glance.i18n import _
from glance.image_cache.drivers import common
def get_db_path():
    """Return the local path to sqlite database."""
    db = CONF.image_cache_sqlite_db
    base_dir = CONF.image_cache_dir
    db_file = os.path.join(base_dir, db)
    if not os.path.exists(db_file):
        LOG.debug('SQLite caching database not located, skipping migration')
        return
    return db_file