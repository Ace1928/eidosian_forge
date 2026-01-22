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
@utils.arg('backup', metavar='<backup>', help='Name or ID of backup to restore.')
@utils.arg('--volume-id', metavar='<volume>', default=None, help=argparse.SUPPRESS)
@utils.arg('--volume', metavar='<volume>', default=None, help='Name or ID of existing volume to which to restore. This is mutually exclusive with --name and takes priority. Default=None.')
@utils.arg('--name', metavar='<name>', default=None, help='Use the name for new volume creation to restore. This is mutually exclusive with --volume (or the deprecated --volume-id) and --volume (or --volume-id) takes priority. Default=None.')
def do_backup_restore(cs, args):
    """Restores a backup."""
    vol = args.volume or args.volume_id
    if vol:
        volume_id = utils.find_volume(cs, vol).id
        if args.name:
            args.name = None
            print('Mutually exclusive options are specified simultaneously: "--volume (or the deprecated --volume-id) and --name". The --volume (or --volume-id) option takes priority.')
    else:
        volume_id = None
    backup = shell_utils.find_backup(cs, args.backup)
    restore = cs.restores.restore(backup.id, volume_id, args.name)
    info = {'backup_id': backup.id}
    info.update(restore._info)
    info.pop('links', None)
    shell_utils.print_dict(info)