import operator
import re
import sqlalchemy as sa
from .. import config
from .. import engines
from .. import eq_
from .. import expect_raises
from .. import expect_raises_message
from .. import expect_warnings
from .. import fixtures
from .. import is_
from ..provision import get_temp_table_name
from ..provision import temp_table_keyword_args
from ..schema import Column
from ..schema import Table
from ... import event
from ... import ForeignKey
from ... import func
from ... import Identity
from ... import inspect
from ... import Integer
from ... import MetaData
from ... import String
from ... import testing
from ... import types as sql_types
from ...engine import Inspector
from ...engine import ObjectKind
from ...engine import ObjectScope
from ...exc import NoSuchTableError
from ...exc import UnreflectableTableError
from ...schema import DDL
from ...schema import Index
from ...sql.elements import quoted_name
from ...sql.schema import BLANK_SCHEMA
from ...testing import ComparesIndexes
from ...testing import ComparesTables
from ...testing import is_false
from ...testing import is_true
from ...testing import mock
def exp_columns(self, schema=None, scope=ObjectScope.ANY, kind=ObjectKind.ANY, filter_names=None):

    def col(name, auto=False, default=mock.ANY, comment=None, nullable=True):
        res = {'name': name, 'autoincrement': auto, 'type': mock.ANY, 'default': default, 'comment': comment, 'nullable': nullable}
        if auto == 'omit':
            res.pop('autoincrement')
        return res

    def pk(name, **kw):
        kw = {'auto': True, 'default': mock.ANY, 'nullable': False, **kw}
        return col(name, **kw)
    materialized = {(schema, 'dingalings_v'): [col('dingaling_id', auto='omit', nullable=mock.ANY), col('address_id'), col('id_user'), col('data')]}
    views = {(schema, 'email_addresses_v'): [col('address_id', auto='omit', nullable=mock.ANY), col('remote_user_id'), col('email_address')], (schema, 'users_v'): [col('user_id', auto='omit', nullable=mock.ANY), col('test1', nullable=mock.ANY), col('test2', nullable=mock.ANY), col('parent_user_id')], (schema, 'user_tmp_v'): [col('id', auto='omit', nullable=mock.ANY), col('name'), col('foo')]}
    self._resolve_views(views, materialized)
    tables = {(schema, 'users'): [pk('user_id'), col('test1', nullable=False), col('test2', nullable=False), col('parent_user_id')], (schema, 'dingalings'): [pk('dingaling_id'), col('address_id'), col('id_user'), col('data')], (schema, 'email_addresses'): [pk('address_id'), col('remote_user_id'), col('email_address')], (schema, 'comment_test'): [pk('id', comment='id comment'), col('data', comment='data % comment'), col('d2', comment='Comment types type speedily \' " \\ \'\' Fun!'), col('d3', comment='Comment\nwith\rescapes')], (schema, 'no_constraints'): [col('data')], (schema, 'local_table'): [pk('id'), col('data'), col('remote_id')], (schema, 'remote_table'): [pk('id'), col('local_id'), col('data')], (schema, 'remote_table_2'): [pk('id'), col('data')], (schema, 'noncol_idx_test_nopk'): [col('q')], (schema, 'noncol_idx_test_pk'): [pk('id'), col('q')], (schema, self.temp_table_name()): [pk('id'), col('name'), col('foo')]}
    res = self._resolve_kind(kind, tables, views, materialized)
    res = self._resolve_names(schema, scope, filter_names, res)
    return res