import os
from alembic import command as alembic_api
from alembic import config as alembic_config
from alembic import migration as alembic_migration
from alembic import script as alembic_script
from oslo_db import exception as db_exception
from oslo_log import log as logging
from oslo_utils import fileutils
from keystone.common import sql
import keystone.conf
def get_current_heads():
    """Get the current head of each the expand and contract branches."""
    config = _find_alembic_conf()
    with sql.session_for_read() as session:
        engine = session.get_bind()
    engine_url = str(engine.url).replace('%', '%%')
    config.set_main_option('sqlalchemy.url', str(engine_url))
    heads = _get_current_heads(engine, config)
    return heads