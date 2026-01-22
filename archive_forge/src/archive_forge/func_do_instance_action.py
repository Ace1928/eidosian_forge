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
@utils.arg('server', metavar='<server>', help=_('Name or UUID of the server to show actions for.'), start_version='2.0', end_version='2.20')
@utils.arg('server', metavar='<server>', help=_('Name or UUID of the server to show actions for. Only UUID can be used to show actions for a deleted server.'), start_version='2.21')
@utils.arg('request_id', metavar='<request_id>', help=_('Request ID of the action to get.'))
def do_instance_action(cs, args):
    """Show an action."""
    if cs.api_version < api_versions.APIVersion('2.21'):
        server = _find_server(cs, args.server)
    else:
        server = _find_server(cs, args.server, raise_if_notfound=False)
    action_resource = cs.instance_action.get(server, args.request_id)
    action = action_resource.to_dict()
    if 'events' in action:
        action['events'] = pprint.pformat(action['events'])
    utils.print_dict(action)