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
def test_paginate_on_hybrid_assert_stmt(self):
    s = Session()
    q = s.query(FakeTable)
    q = utils.paginate_query(q, FakeTable, 5, ['user_id', 'some_hybrid'], sort_dirs=['asc', 'desc'])
    expected_core_sql = select(FakeTable).order_by(sqlalchemy.asc(FakeTable.user_id)).order_by(sqlalchemy.desc(FakeTable.some_hybrid)).limit(5)
    self.assertEqual(str(expected_core_sql.compile()), str(q.statement.compile()))