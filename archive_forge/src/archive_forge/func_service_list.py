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
def service_list(self):
    ctxt = context.get_admin_context()
    services = [service_utils.format_service(service) for service in service_objects.Service.get_all(ctxt)]
    print_format = '%-16s %-16s %-36s %-10s %-10s %-10s %-10s'
    print(print_format % (_('Hostname'), _('Binary'), _('Engine_Id'), _('Host'), _('Topic'), _('Status'), _('Updated At')))
    for svc in services:
        print(print_format % (svc['hostname'], svc['binary'], svc['engine_id'], svc['host'], svc['topic'], svc['status'], svc['updated_at']))