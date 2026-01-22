import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('instance', metavar='<instance>', type=str, help=_('ID or name of the instance.'))
@utils.arg('module', metavar='<module>', type=str, help=_('ID or name of the module.'))
@utils.service_type('database')
def do_module_remove(cs, args):
    """Remove a module from an instance."""
    instance = _find_instance(cs, args.instance)
    module = _find_module(cs, args.module)
    cs.instances.module_remove(instance, module)