import itertools
from unittest import mock
import ddt
import fixtures
from oslo_utils import strutils
from manilaclient import api_versions
from manilaclient import client
from manilaclient.common.apiclient import utils as apiclient_utils
from manilaclient.common import cliutils
from manilaclient.common import constants
from manilaclient import exceptions
from manilaclient import shell
from manilaclient.tests.unit import utils as test_utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient import utils
from manilaclient.v2 import messages
from manilaclient.v2 import security_services
from manilaclient.v2 import share_access_rules
from manilaclient.v2 import share_group_types
from manilaclient.v2 import share_groups
from manilaclient.v2 import share_instances
from manilaclient.v2 import share_network_subnets
from manilaclient.v2 import share_networks
from manilaclient.v2 import share_servers
from manilaclient.v2 import share_snapshots
from manilaclient.v2 import share_types
from manilaclient.v2 import shares
from manilaclient.v2 import shell as shell_v2
@ddt.data([1, 0], [1, 1], [2, 0], [2, 1], [2, 2])
@ddt.unpack
@mock.patch.object(shell_v2, '_find_share_replica', mock.Mock())
def test_share_replica_delete_errors(self, replica_count, replica_errors):

    class StubbedReplicaFindError(Exception):
        """Error in find share replica stub"""
        pass

    class StubbedFindWithErrors(object):

        def __init__(self, existing_replicas):
            self.existing_replicas = existing_replicas

        def __call__(self, cs, replica):
            if replica not in self.existing_replicas:
                raise StubbedReplicaFindError
            return type('FakeShareReplica', (object,), {'id': replica})
    all_replicas = []
    existing_replicas = []
    for counter in range(replica_count):
        replica = 'fake-replica-%d' % counter
        if counter >= replica_errors:
            existing_replicas.append(replica)
        all_replicas.append(replica)
    shell_v2._find_share_replica.side_effect = StubbedFindWithErrors(existing_replicas)
    cmd = 'share-replica-delete %s' % ' '.join(all_replicas)
    if replica_count == replica_errors:
        self.assertRaises(exceptions.CommandError, self.run_command, cmd)
    else:
        self.run_command(cmd)
        for replica in existing_replicas:
            self.assert_called_anytime('DELETE', '/share-replicas/' + replica, clear_callstack=False)