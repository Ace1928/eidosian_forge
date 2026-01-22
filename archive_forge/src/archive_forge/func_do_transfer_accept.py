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
@utils.arg('transfer', metavar='<transfer>', help='ID of transfer to accept.')
@utils.arg('auth_key', metavar='<auth_key>', help='Authentication key of transfer to accept.')
def do_transfer_accept(cs, args):
    """Accepts a volume transfer."""
    transfer = cs.transfers.accept(args.transfer, args.auth_key)
    info = dict()
    info.update(transfer._info)
    info.pop('links', None)
    shell_utils.print_dict(info)