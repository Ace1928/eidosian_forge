import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('instance_id', metavar='<instance_id>', help=_('UUID for instance.'))
@utils.arg('key', metavar='<key>', help=_('Metadata key to delete.'))
@utils.service_type('database')
def do_metadata_delete(cs, args):
    """Deletes metadata for instance <id>."""
    cs.metadata.delete(args.instance_id, args.key)