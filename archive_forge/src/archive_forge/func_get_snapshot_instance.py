import ast
import re
import time
from oslo_utils import strutils
from tempest.lib.cli import base
from tempest.lib.cli import output_parser
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions as tempest_lib_exc
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import exceptions
from manilaclient.tests.functional import utils
def get_snapshot_instance(self, id=None, microversion=None):
    """Get snapshot instance."""
    cmd = 'snapshot-instance-show %s ' % id
    snapshot_instance_raw = self.manila(cmd, microversion=microversion)
    snapshot_instance = output_parser.details(snapshot_instance_raw)
    return snapshot_instance