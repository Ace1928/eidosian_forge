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
def test_passes_other_exceptions(self):
    with testtools.ExpectedException(ValueError):
        with db_api.exc_to_retry(TypeError):
            raise ValueError()