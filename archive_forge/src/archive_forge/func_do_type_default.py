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
def do_type_default(cs, args):
    """List the default volume type.

    The Block Storage service allows configuration of a default
    type for each project, as well as the system default, so use
    this command to determine what your effective default volume
    type is.
    """
    vtype = cs.volume_types.default()
    shell_utils.print_volume_type_list([vtype])