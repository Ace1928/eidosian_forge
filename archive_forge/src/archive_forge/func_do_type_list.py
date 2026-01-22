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
def do_type_list(cs, args):
    """Lists available 'volume types'.

    (Only admin and tenant users will see private types)
    """
    vtypes = cs.volume_types.list()
    shell_utils.print_volume_type_list(vtypes)