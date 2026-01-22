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
@api_versions.wraps('2.26')
@utils.arg('server', metavar='<server>', help=_('Name or ID of server.'))
@utils.arg('tag', metavar='<tag>', nargs='+', help=_('Tag(s) to add.'))
def do_server_tag_add(cs, args):
    """Add one or more tags to a server."""
    server = _find_server(cs, args.server)
    utils.do_action_on_many(lambda t: server.add_tag(t), args.tag, _('Request to add tag %s to specified server has been accepted.'), _('Unable to add the specified tag to the server.'))