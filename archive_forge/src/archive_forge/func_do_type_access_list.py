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
@utils.arg('--volume-type', metavar='<volume_type>', required=True, help='Filter results by volume type name or ID.')
def do_type_access_list(cs, args):
    """Print access information about the given volume type."""
    volume_type = shell_utils.find_volume_type(cs, args.volume_type)
    if volume_type.is_public:
        raise exceptions.CommandError('Failed to get access list for public volume type.')
    access_list = cs.volume_type_access.list(volume_type)
    columns = ['Volume_type_ID', 'Project_ID']
    shell_utils.print_list(access_list, columns)