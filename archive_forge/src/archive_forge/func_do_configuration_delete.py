import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('configuration_group', metavar='<configuration_group>', help=_('ID or name of the configuration group.'))
@utils.service_type('database')
def do_configuration_delete(cs, args):
    """Deletes a configuration group."""
    configuration = _find_configuration(cs, args.configuration_group)
    cs.configurations.delete(configuration)