import datetime
from unittest import mock
import uuid
import freezegun
import http.client
from oslo_db import exception as oslo_db_exception
from oslo_utils import timeutils
from testtools import matchers
from keystone.common import provider_api
from keystone.common import utils
from keystone.models import revoke_model
from keystone.tests.unit import test_v3
def test_retries_on_deadlock(self):
    patcher = mock.patch('sqlalchemy.orm.query.Query.delete', autospec=True)

    class FakeDeadlock(object):

        def __init__(self, mock_patcher):
            self.deadlock_count = 2
            self.mock_patcher = mock_patcher
            self.patched = True

        def __call__(self, *args, **kwargs):
            if self.deadlock_count > 1:
                self.deadlock_count -= 1
            else:
                self.mock_patcher.stop()
                self.patched = False
            raise oslo_db_exception.DBDeadlock
    sql_delete_mock = patcher.start()
    side_effect = FakeDeadlock(patcher)
    sql_delete_mock.side_effect = side_effect
    try:
        PROVIDERS.revoke_api.revoke(revoke_model.RevokeEvent(user_id=uuid.uuid4().hex))
    finally:
        if side_effect.patched:
            patcher.stop()
    call_count = sql_delete_mock.call_count
    revoke_attempt_count = 2
    self.assertEqual(call_count, revoke_attempt_count)