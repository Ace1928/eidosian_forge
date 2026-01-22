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
def exp_ccs(self, schema=None, scope=ObjectScope.ANY, kind=ObjectKind.ANY, filter_names=None):

    class tt(str):

        def __eq__(self, other):
            res = other.lower().replace('(', '').replace(')', '').replace('`', '')
            return self in res

    def cc(text, name, comment=None):
        return {'sqltext': tt(text), 'name': name, 'comment': comment}
    materialized = {(schema, 'dingalings_v'): []}
    views = {(schema, 'email_addresses_v'): [], (schema, 'users_v'): [], (schema, 'user_tmp_v'): []}
    self._resolve_views(views, materialized)
    tables = {(schema, 'users'): [cc('test2 <= 1000', mock.ANY), cc('test2 > 0', 'zz_test2_gt_zero', comment='users check constraint')], (schema, 'dingalings'): [cc('address_id > 0 and address_id < 1000', name='address_id_gt_zero')], (schema, 'email_addresses'): [], (schema, 'comment_test'): [], (schema, 'no_constraints'): [], (schema, 'local_table'): [], (schema, 'remote_table'): [], (schema, 'remote_table_2'): [], (schema, 'noncol_idx_test_nopk'): [], (schema, 'noncol_idx_test_pk'): [], (schema, self.temp_table_name()): []}
    res = self._resolve_kind(kind, tables, views, materialized)
    res = self._resolve_names(schema, scope, filter_names, res)
    return res