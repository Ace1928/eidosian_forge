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
def do_revision(config, cmd):
    kwargs = {'message': CONF.command.message, 'autogenerate': CONF.command.autogenerate, 'sql': CONF.command.sql}
    branches = []
    if CONF.command.expand:
        kwargs['head'] = 'expand@head'
        branches.append(upgrades.EXPAND_BRANCH)
    elif CONF.command.contract:
        kwargs['head'] = 'contract@head'
        branches.append(upgrades.CONTRACT_BRANCH)
    else:
        branches = upgrades.MIGRATION_BRANCHES
    if not CONF.command.autogenerate:
        for branch in branches:
            args = copy.copy(kwargs)
            version_path = upgrades.get_version_branch_path(release=upgrades.CURRENT_RELEASE, branch=branch)
            upgrades.check_bootstrap_new_branch(branch, version_path, args)
            do_alembic_command(config, cmd, **args)
    else:
        do_alembic_command(config, cmd, **kwargs)