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
def exp_pks(self, schema=None, scope=ObjectScope.ANY, kind=ObjectKind.ANY, filter_names=None):

    def pk(*cols, name=mock.ANY, comment=None):
        return {'constrained_columns': list(cols), 'name': name, 'comment': comment}
    empty = pk(name=None)
    if testing.requires.materialized_views_reflect_pk.enabled:
        materialized = {(schema, 'dingalings_v'): pk('dingaling_id')}
    else:
        materialized = {(schema, 'dingalings_v'): empty}
    views = {(schema, 'email_addresses_v'): empty, (schema, 'users_v'): empty, (schema, 'user_tmp_v'): empty}
    self._resolve_views(views, materialized)
    tables = {(schema, 'users'): pk('user_id'), (schema, 'dingalings'): pk('dingaling_id'), (schema, 'email_addresses'): pk('address_id', name='email_ad_pk', comment='ea pk comment'), (schema, 'comment_test'): pk('id'), (schema, 'no_constraints'): empty, (schema, 'local_table'): pk('id'), (schema, 'remote_table'): pk('id'), (schema, 'remote_table_2'): pk('id'), (schema, 'noncol_idx_test_nopk'): empty, (schema, 'noncol_idx_test_pk'): pk('id'), (schema, self.temp_table_name()): pk('id')}
    if not testing.requires.reflects_pk_names.enabled:
        for val in tables.values():
            if val['name'] is not None:
                val['name'] = mock.ANY
    res = self._resolve_kind(kind, tables, views, materialized)
    res = self._resolve_names(schema, scope, filter_names, res)
    return res