import argparse
import collections
import copy
import os
from oslo_utils import strutils
from cinderclient import base
from cinderclient import exceptions
from cinderclient import shell_utils
from cinderclient import utils
from cinderclient.v3 import availability_zones
@utils.arg('host', metavar='<host>', help='Cinder host on which the existing volume resides; takes the form: host@backend-name#pool')
@utils.arg('identifier', metavar='<identifier>', help='Name or other Identifier for existing volume')
@utils.arg('--id-type', metavar='<id-type>', default='source-name', help='Type of backend device identifier provided, typically source-name or source-id (Default=source-name)')
@utils.arg('--name', metavar='<name>', help='Volume name (Default=None)')
@utils.arg('--description', metavar='<description>', help='Volume description (Default=None)')
@utils.arg('--volume-type', metavar='<volume-type>', help='Volume type (Default=None)')
@utils.arg('--availability-zone', metavar='<availability-zone>', help='Availability zone for volume (Default=None)')
@utils.arg('--metadata', nargs='*', metavar='<key=value>', help='Metadata key=value pairs (Default=None)')
@utils.arg('--bootable', action='store_true', help='Specifies that the newly created volume should be marked as bootable')
def do_manage(cs, args):
    """Manage an existing volume."""
    volume_metadata = None
    if args.metadata is not None:
        volume_metadata = shell_utils.extract_metadata(args)
    ref_dict = {args.id_type: args.identifier}
    if hasattr(args, 'source_name') and args.source_name is not None:
        ref_dict['source-name'] = args.source_name
    if hasattr(args, 'source_id') and args.source_id is not None:
        ref_dict['source-id'] = args.source_id
    volume = cs.volumes.manage(host=args.host, ref=ref_dict, name=args.name, description=args.description, volume_type=args.volume_type, availability_zone=args.availability_zone, metadata=volume_metadata, bootable=args.bootable)
    info = {}
    volume = cs.volumes.get(volume.id)
    info.update(volume._info)
    info.pop('links', None)
    shell_utils.print_dict(info)