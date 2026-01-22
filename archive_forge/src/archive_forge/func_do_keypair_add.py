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
@utils.arg('name', metavar='<name>', help=_('Name of key.'))
@utils.arg('--pub-key', metavar='<pub-key>', default=None, help=_('Path to a public ssh key.'))
@utils.arg('--key-type', metavar='<key-type>', default='ssh', help=_('Keypair type. Can be ssh or x509.'), start_version='2.2')
@utils.arg('--user', metavar='<user-id>', default=None, help=_('ID of user to whom to add key-pair (Admin only).'), start_version='2.10')
def do_keypair_add(cs, args):
    """Create a new key pair for use with servers."""
    name = args.name
    pub_key = args.pub_key
    if pub_key:
        if pub_key == '-':
            pub_key = sys.stdin.read()
        else:
            try:
                with open(os.path.expanduser(pub_key)) as f:
                    pub_key = f.read()
            except IOError as e:
                raise exceptions.CommandError(_("Can't open or read '%(key)s': %(exc)s") % {'key': pub_key, 'exc': e})
    keypair = _keypair_create(cs, args, name, pub_key)
    if not pub_key:
        private_key = keypair.private_key
        print(private_key)