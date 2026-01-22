import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('instance', metavar='<instance>', type=str, help=_('ID or name of the instance.'))
@utils.arg('configuration', metavar='<configuration>', type=str, help=_('ID or name of the configuration group to attach to the instance.'))
@utils.service_type('database')
def do_configuration_attach(cs, args):
    """Attaches a configuration group to an instance."""
    instance = _find_instance(cs, args.instance)
    configuration = _find_configuration(cs, args.configuration)
    cs.instances.modify(instance, configuration)