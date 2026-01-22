import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('--datastore', metavar='<datastore>', help=_("Name or ID of datastore to list modules for. Use '%s' to list modules that apply to all datastores.") % modules.Module.ALL_KEYWORD)
@utils.service_type('database')
def do_module_list(cs, args):
    """Lists the modules available."""
    datastore = None
    if args.datastore:
        if args.datastore.lower() == modules.Module.ALL_KEYWORD:
            datastore = args.datastore.lower()
        else:
            datastore = _find_datastore(cs, args.datastore)
    module_list = cs.modules.list(datastore=datastore)
    field_list = ['id', 'name', 'type', 'datastore', 'datastore_version', 'auto_apply', 'priority_apply', 'apply_order', 'is_admin', 'tenant', 'visible']
    if not utils.is_admin(cs):
        field_list = field_list[:-2]
    utils.print_list(module_list, field_list, labels={'datastore_version': 'Version', 'priority_apply': 'Priority', 'apply_order': 'Order', 'is_admin': 'Admin'})