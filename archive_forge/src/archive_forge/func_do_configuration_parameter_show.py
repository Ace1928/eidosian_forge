import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('--datastore', metavar='<datastore>', default=None, help=_('ID or name of the datastore to list configuration parameters for. Optional if the ID of the datastore_version is provided.'))
@utils.arg('datastore_version', metavar='<datastore_version>', help=_('Datastore version name or ID assigned to the configuration group.'))
@utils.arg('parameter', metavar='<parameter>', help=_('Name of the configuration parameter.'))
@utils.service_type('database')
def do_configuration_parameter_show(cs, args):
    """Shows details of a configuration parameter."""
    if args.datastore:
        param = cs.configuration_parameters.get_parameter(args.datastore, args.datastore_version, args.parameter)
    elif utils.is_uuid_like(args.datastore_version):
        param = cs.configuration_parameters.get_parameter_by_version(args.datastore_version, args.parameter)
    else:
        raise exceptions.NoUniqueMatch(_('The datastore name or id is required to retrieve the parameter for the configuration group by name.'))
    _print_object(param)