import copy
import sys
from alembic import command as alembic_command
from alembic import script as alembic_script
from alembic import util as alembic_util
from oslo_config import cfg
from oslo_log import log
import pbr.version
from keystone.common import sql
from keystone.common.sql import upgrades
import keystone.conf
from keystone.i18n import _
def do_generic_show(config, cmd):
    kwargs = {'verbose': CONF.command.verbose}
    do_alembic_command(config, cmd, **kwargs)