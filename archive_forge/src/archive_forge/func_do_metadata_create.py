import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('instance_id', metavar='<instance_id>', help=_('UUID for instance.'))
@utils.arg('key', metavar='<key>', help=_('Key for assignment.'))
@utils.arg('value', metavar='<value>', help=_('Value to assign to <key>.'))
@utils.service_type('database')
def do_metadata_create(cs, args):
    """Creates metadata in the database for instance <id>."""
    result = cs.metadata.create(args.instance_id, args.key, args.value)
    _print_object(result)