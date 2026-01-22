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
def do_resource_data_list():
    ctxt = context.get_admin_context()
    data = db_api.resource_data_get_all(ctxt, CONF.command.resource_id)
    print_format = '%-16s %-64s'
    for k in data.keys():
        print(print_format % (k, data[k]))