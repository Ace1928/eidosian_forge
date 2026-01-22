import argparse
import collections
import datetime
import getpass
import logging
import os
import pprint
import sys
import time
from oslo_utils import netutils
from oslo_utils import strutils
from oslo_utils import timeutils
import novaclient
from novaclient import api_versions
from novaclient import base
from novaclient import client
from novaclient import exceptions
from novaclient.i18n import _
from novaclient import shell
from novaclient import utils
from novaclient.v2 import availability_zones
from novaclient.v2 import quotas
from novaclient.v2 import servers
@utils.arg('server', metavar='<server>', help=_('Name or ID of server.'))
@utils.arg('volume', metavar='<volume>', help=_('ID of the volume to attach.'))
@utils.arg('device', metavar='<device>', default=None, nargs='?', help=_('Name of the device e.g. /dev/vdb. Use "auto" for autoassign (if supported). Libvirt driver will use default device name.'))
@utils.arg('--tag', metavar='<tag>', default=None, help=_('Tag for the attached volume.'), start_version='2.49')
@utils.arg('--delete-on-termination', action='store_true', default=False, help=_('Specify if the attached volume should be deleted when the server is destroyed.'), start_version='2.79')
def do_volume_attach(cs, args):
    """Attach a volume to a server."""
    if args.device == 'auto':
        args.device = None
    update_kwargs = {}
    if 'tag' in args and args.tag:
        update_kwargs['tag'] = args.tag
    if 'delete_on_termination' in args and args.delete_on_termination:
        update_kwargs['delete_on_termination'] = args.delete_on_termination
    volume = cs.volumes.create_server_volume(_find_server(cs, args.server).id, args.volume, args.device, **update_kwargs)
    _print_volume(volume)