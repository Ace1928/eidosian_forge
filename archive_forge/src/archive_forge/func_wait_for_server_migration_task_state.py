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
def wait_for_server_migration_task_state(self, share_server_id, dest_host, task_state_to_wait, microversion=None):
    """Waits for a certain server task state. """
    statuses = (task_state_to_wait,) if not isinstance(task_state_to_wait, (tuple, list, set)) else task_state_to_wait
    server = self.get_share_server(share_server=share_server_id, microversion=microversion)
    start = int(time.time())
    while server['task_state'] not in statuses:
        time.sleep(self.build_interval)
        server = self.get_share_server(share_server=share_server_id, microversion=microversion)
        if server['task_state'] in statuses:
            return server
        elif server['task_state'] == constants.TASK_STATE_MIGRATION_ERROR:
            raise exceptions.ShareServerMigrationException(server_id=server['id'])
        elif int(time.time()) - start >= self.build_timeout:
            message = 'Server %(share_server_id)s failed to reach the status in %(status)s while migrating from host %(src)s to host %(dest)s within the required time %(timeout)s.' % {'src': server['host'], 'dest': dest_host, 'share_server_id': server['id'], 'timeout': self.build_timeout, 'status': str(statuses)}
            raise tempest_lib_exc.TimeoutException(message)