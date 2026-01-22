from unittest import mock
from urllib import parse
import fixtures
import sqlalchemy
from sqlalchemy import Boolean, Index, Integer, DateTime, String
from sqlalchemy import MetaData, Table, Column
from sqlalchemy import ForeignKey, ForeignKeyConstraint
from sqlalchemy.dialects.postgresql import psycopg2
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import column_property
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import registry
from sqlalchemy.orm import Session
from sqlalchemy import sql
from sqlalchemy.sql.expression import cast
from sqlalchemy.sql import select
from sqlalchemy.types import UserDefinedType
from oslo_db import exception
from oslo_db.sqlalchemy import models
from oslo_db.sqlalchemy import provision
from oslo_db.sqlalchemy import session
from oslo_db.sqlalchemy import utils
from oslo_db.tests import base as test_base
from oslo_db.tests.sqlalchemy import base as db_test_base
def test_ensure_backend_available_no_dbapi_raises(self):
    log = self.useFixture(fixtures.FakeLogger())
    with mock.patch.object(sqlalchemy, 'create_engine') as mock_create:
        mock_create.side_effect = ImportError("Can't import DBAPI module foobar")
        exc = self.assertRaises(exception.BackendNotAvailable, provision.Backend._ensure_backend_available, self.connect_string)
        mock_create.assert_called_once_with(utils.make_url(self.connect_string))
        self.assertEqual("Backend 'postgresql+psycopg2' is unavailable: No DBAPI installed", str(exc))
        self.assertEqual("The postgresql+psycopg2 backend is unavailable: Can't import DBAPI module foobar", log.output.strip())