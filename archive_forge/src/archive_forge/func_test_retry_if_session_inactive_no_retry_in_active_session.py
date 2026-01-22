from unittest import mock
from oslo_db import exception as db_exc
import osprofiler
import sqlalchemy
from sqlalchemy.orm import exc
import testtools
from neutron_lib.db import api as db_api
from neutron_lib import exceptions
from neutron_lib import fixture
from neutron_lib.tests import _base
def test_retry_if_session_inactive_no_retry_in_active_session(self):
    context = mock.Mock()
    context.session.is_active = True
    with testtools.ExpectedException(db_exc.DBDeadlock):
        self._context_function(context, [], {1: 2}, fail_count=1, exc_to_raise=db_exc.DBDeadlock())