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
def test_drop_old_duplicate_entries_from_table(self):
    table_name = '__test_tmp_table__'
    test_table, values = self._populate_db_for_drop_duplicate_entries(self.engine, self.meta, table_name)
    utils.drop_old_duplicate_entries_from_table(self.engine, table_name, False, 'b', 'c')
    uniq_values = set()
    expected_ids = []
    for value in sorted(values, key=lambda x: x['id'], reverse=True):
        uniq_value = (('b', value['b']), ('c', value['c']))
        if uniq_value in uniq_values:
            continue
        uniq_values.add(uniq_value)
        expected_ids.append(value['id'])
    with self.engine.connect() as conn, conn.begin():
        real_ids = [row[0] for row in conn.execute(select(test_table.c.id)).fetchall()]
    self.assertEqual(len(expected_ids), len(real_ids))
    for id_ in expected_ids:
        self.assertIn(id_, real_ids)