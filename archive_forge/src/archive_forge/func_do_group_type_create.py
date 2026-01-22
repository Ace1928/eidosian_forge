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
@utils.arg('name', metavar='<name>', help='Name of new group type.')
@utils.arg('--description', metavar='<description>', help='Description of new group type.')
@utils.arg('--is-public', metavar='<is-public>', default=True, help='Make type accessible to the public (default true).')
def do_group_type_create(cs, args):
    """Creates a group type."""
    is_public = strutils.bool_from_string(args.is_public)
    gtype = cs.group_types.create(args.name, args.description, is_public)
    shell_utils.print_group_type_list([gtype])