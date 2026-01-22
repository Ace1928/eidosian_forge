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
def list_snapshot_instances(self, snapshot_id=None, columns=None, detailed=None, microversion=None):
    """List snapshot instances."""
    cmd = 'snapshot-instance-list '
    if snapshot_id:
        cmd += '--snapshot %s' % snapshot_id
    if columns is not None:
        cmd += ' --columns ' + columns
    if detailed:
        cmd += ' --detailed True '
    snapshot_instances_raw = self.manila(cmd, microversion=microversion)
    snapshot_instances = utils.listing(snapshot_instances_raw)
    return snapshot_instances