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
@mock.patch.object(sqlalchemy.sql, 'and_')
@mock.patch.object(sqlalchemy.sql, 'or_')
def test_paginate_query(self, mock_or, mock_and):
    self.query.order_by.return_value = self.query
    self.query.filter.return_value = self.query
    self.mock_asc.return_value = 'asc_1'
    self.mock_desc.return_value = 'desc_1'
    mock_and.side_effect = ['some_crit', 'another_crit']
    mock_or.return_value = 'some_f'
    utils.paginate_query(self.query, self.model, 5, ['user_id', 'project_id'], marker=self.marker, sort_dirs=['asc', 'desc'])
    self.mock_asc.assert_called_once_with(self.model.user_id)
    self.mock_desc.assert_called_once_with(self.model.project_id)
    self.query.order_by.assert_has_calls([mock.call('asc_1'), mock.call('desc_1')])
    mock_and.assert_has_calls([mock.call(mock.ANY), mock.call(mock.ANY, mock.ANY)])
    mock_or.assert_called_once_with('some_crit', 'another_crit')
    self.query.filter.assert_called_once_with('some_f')
    self.query.limit.assert_called_once_with(5)