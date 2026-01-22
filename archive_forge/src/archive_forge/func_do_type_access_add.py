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
@utils.arg('--volume-type', metavar='<volume_type>', required=True, help='Volume type name or ID to add access for the given project.')
@utils.arg('--project-id', metavar='<project_id>', required=True, help='Project ID to add volume type access for.')
def do_type_access_add(cs, args):
    """Adds volume type access for the given project."""
    vtype = shell_utils.find_volume_type(cs, args.volume_type)
    cs.volume_type_access.add_project_access(vtype, args.project_id)