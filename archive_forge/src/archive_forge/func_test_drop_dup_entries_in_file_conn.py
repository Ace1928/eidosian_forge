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
def test_drop_dup_entries_in_file_conn(self):
    table_name = '__test_tmp_table__'
    tmp_db_file = self.create_tempfiles([['name', '']], ext='.sql')[0]
    in_file_engine = session.EngineFacade('sqlite:///%s' % tmp_db_file).get_engine()
    meta = MetaData()
    test_table, values = self._populate_db_for_drop_duplicate_entries(in_file_engine, meta, table_name)
    utils.drop_old_duplicate_entries_from_table(in_file_engine, table_name, False, 'b', 'c')