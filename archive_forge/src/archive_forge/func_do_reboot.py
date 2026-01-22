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
@utils.arg('--hard', dest='reboot_type', action='store_const', const=servers.REBOOT_HARD, default=servers.REBOOT_SOFT, help=_('Perform a hard reboot (instead of a soft one). Note: Ironic does not currently support soft reboot; consequently, bare metal nodes will always do a hard reboot, regardless of the use of this option.'))
@utils.arg('server', metavar='<server>', nargs='+', help=_('Name or ID of server(s).'))
@utils.arg('--poll', dest='poll', action='store_true', default=False, help=_('Poll until reboot is complete.'))
def do_reboot(cs, args):
    """Reboot a server."""
    servers = [_find_server(cs, s) for s in args.server]
    utils.do_action_on_many(lambda s: s.reboot(args.reboot_type), servers, _('Request to reboot server %s has been accepted.'), _('Unable to reboot the specified server(s).'))
    if args.poll:
        utils.do_action_on_many(lambda s: _poll_for_status(cs.servers.get, s.id, 'rebooting', ['active'], show_progress=False), servers, _('Wait for server %s reboot.'), _('Wait for specified server(s) failed.'))