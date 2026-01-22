import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('configuration_group', metavar='<configuration_group>', help=_('ID or name of the configuration group.'))
@utils.arg('values', metavar='<values>', help=_('Dictionary of the values to set.'))
@utils.service_type('database')
def do_configuration_patch(cs, args):
    """Patches a configuration group."""
    configuration = _find_configuration(cs, args.configuration_group)
    cs.configurations.edit(configuration, args.values)