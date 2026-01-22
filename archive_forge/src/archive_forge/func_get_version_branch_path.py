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
def get_version_branch_path(release=None, branch=None):
    """Get the path to a version branch."""
    version_path = VERSIONS_PATH
    if branch and release:
        return os.path.join(version_path, release, branch)
    return version_path