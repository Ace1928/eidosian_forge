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
@utils.arg('class_name', metavar='<class>', help='Name of quota class for which to list quotas.')
def do_quota_class_show(cs, args):
    """Lists quotas for a quota class."""
    shell_utils.quota_show(cs.quota_classes.get(args.class_name))