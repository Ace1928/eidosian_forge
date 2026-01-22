import sys
from oslo_config import cfg
from oslo_log import log
from heat.common import context
from heat.common import exception
from heat.common.i18n import _
from heat.common import messaging
from heat.common import service_utils
from heat.db import api as db_api
from heat.db import migration as db_migration
from heat.objects import service as service_objects
from heat.rpc import client as rpc_client
from heat import version
def do_migrate():
    messaging.setup()
    client = rpc_client.EngineClient()
    ctxt = context.get_admin_context()
    try:
        client.migrate_convergence_1(ctxt, CONF.command.stack_id)
    except exception.NotFound:
        raise Exception(_('Stack with id %s can not be found.') % CONF.command.stack_id)
    except (exception.NotSupported, exception.ActionNotComplete) as ex:
        raise Exception(ex.message)