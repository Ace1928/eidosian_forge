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
@utils.arg('vtype', metavar='<vtype>', help='Name or ID of volume type.')
@utils.arg('action', metavar='<action>', choices=['set', 'unset'], help='The action. Valid values are "set" or "unset."')
@utils.arg('metadata', metavar='<key=value>', nargs='+', default=[], help='The extra specs key and value pair to set or unset. For unset, specify only the key.')
def do_type_key(cs, args):
    """Sets or unsets extra_spec for a volume type."""
    vtype = shell_utils.find_volume_type(cs, args.vtype)
    keypair = shell_utils.extract_metadata(args)
    if args.action == 'set':
        vtype.set_keys(keypair)
    elif args.action == 'unset':
        vtype.unset_keys(list(keypair))