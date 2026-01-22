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
@utils.arg('volume_type', metavar='<volume_type>', help='Name or ID of the volume type.')
def do_type_show(cs, args):
    """Show volume type details."""
    vtype = shell_utils.find_vtype(cs, args.volume_type)
    info = dict()
    info.update(vtype._info)
    info.pop('links', None)
    shell_utils.print_dict(info, formatters=['extra_specs'])