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
def test_inner_exception_preserved_in_retryrequest(self):
    try:
        exc = ValueError('test')
        with db_api.exc_to_retry(ValueError):
            raise exc
    except db_exc.RetryRequest as e:
        self.assertEqual(exc, e.inner_exc)