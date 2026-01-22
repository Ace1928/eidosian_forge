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
def test_multi_exception_raised_on_exceed(self):
    retry_fixture = fixture.DBRetryErrorsFixture(max_retries=2)
    retry_fixture.setUp()
    e = exceptions.MultipleExceptions([ValueError(), db_exc.DBDeadlock()])
    with testtools.ExpectedException(exceptions.MultipleExceptions):
        self._decorated_function(db_api.MAX_RETRIES + 1, e)
    retry_fixture.cleanUp()