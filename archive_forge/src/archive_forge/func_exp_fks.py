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
def exp_fks(self, schema=None, scope=ObjectScope.ANY, kind=ObjectKind.ANY, filter_names=None):

    class tt:

        def __eq__(self, other):
            return other is None or config.db.dialect.default_schema_name == other

    def fk(cols, ref_col, ref_table, ref_schema=schema, name=mock.ANY, comment=None):
        return {'constrained_columns': cols, 'referred_columns': ref_col, 'name': name, 'options': mock.ANY, 'referred_schema': ref_schema if ref_schema is not None else tt(), 'referred_table': ref_table, 'comment': comment}
    materialized = {(schema, 'dingalings_v'): []}
    views = {(schema, 'email_addresses_v'): [], (schema, 'users_v'): [], (schema, 'user_tmp_v'): []}
    self._resolve_views(views, materialized)
    tables = {(schema, 'users'): [fk(['parent_user_id'], ['user_id'], 'users', name='user_id_fk')], (schema, 'dingalings'): [fk(['id_user'], ['user_id'], 'users'), fk(['address_id'], ['address_id'], 'email_addresses', name='zz_email_add_id_fg', comment='di fk comment')], (schema, 'email_addresses'): [fk(['remote_user_id'], ['user_id'], 'users')], (schema, 'comment_test'): [], (schema, 'no_constraints'): [], (schema, 'local_table'): [fk(['remote_id'], ['id'], 'remote_table_2', ref_schema=config.test_schema)], (schema, 'remote_table'): [fk(['local_id'], ['id'], 'local_table', ref_schema=None)], (schema, 'remote_table_2'): [], (schema, 'noncol_idx_test_nopk'): [], (schema, 'noncol_idx_test_pk'): [], (schema, self.temp_table_name()): []}
    if not testing.requires.self_referential_foreign_keys.enabled:
        tables[schema, 'users'].clear()
    if not testing.requires.named_constraints.enabled:
        for vals in tables.values():
            for val in vals:
                if val['name'] is not mock.ANY:
                    val['name'] = mock.ANY
    res = self._resolve_kind(kind, tables, views, materialized)
    res = self._resolve_names(schema, scope, filter_names, res)
    return res