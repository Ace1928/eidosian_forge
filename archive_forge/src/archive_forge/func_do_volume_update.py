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
@utils.arg('src_volume', metavar='<src_volume>', help=_('ID of the source (original) volume.'))
@utils.arg('dest_volume', metavar='<dest_volume>', help=_('ID of the destination volume.'))
@utils.arg('--delete-on-termination', default=None, group='delete_on_termination', action='store_true', help=_('Specify that the volume should be deleted when the server is destroyed.'), start_version='2.85')
@utils.arg('--no-delete-on-termination', group='delete_on_termination', action='store_false', help=_('Specify that the volume should not be deleted when the server is destroyed.'), start_version='2.85')
def do_volume_update(cs, args):
    """Update the attachment on the server.

    If dest_volume is the same as the src_volume then the command migrates
    the data from the attached volume to the specified available volume
    and swaps out the active attachment to the new volume. Otherwise it
    only updates the parameters of the existing attachment.
    """
    kwargs = dict()
    if cs.api_version >= api_versions.APIVersion('2.85') and args.delete_on_termination is not None:
        kwargs['delete_on_termination'] = args.delete_on_termination
    cs.volumes.update_server_volume(_find_server(cs, args.server).id, args.src_volume, args.dest_volume, **kwargs)