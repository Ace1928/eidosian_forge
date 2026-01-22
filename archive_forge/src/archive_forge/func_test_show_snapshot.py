import copy
from unittest import mock
from oslo_messaging._drivers import common as rpc_common
from oslo_utils import reflection
from heat.common import exception
from heat.common import identifier
from heat.rpc import client as rpc_client
from heat.tests import common
from heat.tests import utils
def test_show_snapshot(self):
    snapshot_id = '86729f02-4648-44d8-af44-d0ec65b6abc9'
    self._test_engine_api('show_snapshot', 'call', stack_identity=self.identity, snapshot_id=snapshot_id)