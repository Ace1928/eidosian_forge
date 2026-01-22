import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('instance', metavar='<instance>', type=str, help=_('ID or name of the instance.'))
@utils.service_type('database')
def do_module_list_instance(cs, args):
    """Lists the modules that have been applied to an instance."""
    instance = _find_instance(cs, args.instance)
    module_list = cs.instances.modules(instance)
    utils.print_list(module_list, ['id', 'name', 'type', 'md5', 'created', 'updated'])