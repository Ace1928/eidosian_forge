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
def wait_for_snapshot_status(self, snapshot, status, microversion=None):
    """Waits for a snapshot to reach a given status."""
    body = self.get_snapshot(snapshot, microversion=microversion)
    snapshot_name = body['name']
    snapshot_status = body['status']
    start = int(time.time())
    while snapshot_status != status:
        time.sleep(self.build_interval)
        body = self.get_snapshot(snapshot, microversion=microversion)
        snapshot_status = body['status']
        if snapshot_status == status:
            return
        elif 'error' in snapshot_status.lower():
            raise exceptions.SnapshotBuildErrorException(snapshot=snapshot)
        if int(time.time()) - start >= self.build_timeout:
            message = 'Snapshot %(snapshot_name)s failed to reach %(status)s status within the required time (%(timeout)s s).' % {'snapshot_name': snapshot_name, 'status': status, 'timeout': self.build_timeout}
            raise tempest_lib_exc.TimeoutException(message)