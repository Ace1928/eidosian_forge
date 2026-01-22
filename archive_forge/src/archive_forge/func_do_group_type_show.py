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
@utils.arg('group_type', metavar='<group_type>', help='Name or ID of the group type.')
def do_group_type_show(cs, args):
    """Show group type details."""
    gtype = shell_utils.find_gtype(cs, args.group_type)
    info = dict()
    info.update(gtype._info)
    info.pop('links', None)
    shell_utils.print_dict(info, formatters=['group_specs'])