import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('--datastore_type', metavar='<datastore_type>', default=None, help='Type of the datastore. For eg: mysql.')
@utils.arg('--datastore_version_id', metavar='<datastore_version_id>', default=None, help='ID of the datastore version.')
@utils.service_type('database')
def do_volume_type_list(cs, args):
    """Lists available volume types."""
    if args.datastore_type and args.datastore_version_id:
        volume_types = cs.volume_types.list_datastore_version_associated_volume_types(args.datastore_type, args.datastore_version_id)
    elif not args.datastore_type and (not args.datastore_version_id):
        volume_types = cs.volume_types.list()
    else:
        raise exceptions.MissingArgs(['datastore_type', 'datastore_version_id'])
    utils.print_list(volume_types, ['id', 'name', 'is_public', 'description'])