import argparse
import collections
import os
from oslo_utils import strutils
import cinderclient
from cinderclient import api_versions
from cinderclient import base
from cinderclient import exceptions
from cinderclient import shell_utils
from cinderclient import utils
from cinderclient.v3.shell_base import *  # noqa
from cinderclient.v3.shell_base import CheckSizeArgForCreate
@api_versions.wraps('3.11')
@utils.arg('gtype', metavar='<gtype>', help='Name or ID of group type.')
@utils.arg('action', metavar='<action>', choices=['set', 'unset'], help='The action. Valid values are "set" or "unset."')
@utils.arg('metadata', metavar='<key=value>', nargs='+', default=[], help='The group specs key and value pair to set or unset. For unset, specify only the key.')
def do_group_type_key(cs, args):
    """Sets or unsets group_spec for a group type."""
    gtype = shell_utils.find_group_type(cs, args.gtype)
    keypair = shell_utils.extract_metadata(args)
    if args.action == 'set':
        gtype.set_keys(keypair)
    elif args.action == 'unset':
        gtype.unset_keys(list(keypair))