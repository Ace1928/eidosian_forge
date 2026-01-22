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
@classmethod
def define_reflected_tables(cls, metadata, schema):
    if schema:
        schema_prefix = schema + '.'
    else:
        schema_prefix = ''
    if testing.requires.self_referential_foreign_keys.enabled:
        parent_id_args = (ForeignKey('%susers.user_id' % schema_prefix, name='user_id_fk'),)
    else:
        parent_id_args = ()
    users = Table('users', metadata, Column('user_id', sa.INT, primary_key=True), Column('test1', sa.CHAR(5), nullable=False), Column('test2', sa.Float(), nullable=False), Column('parent_user_id', sa.Integer, *parent_id_args), sa.CheckConstraint('test2 > 0', name='zz_test2_gt_zero', comment='users check constraint'), sa.CheckConstraint('test2 <= 1000'), schema=schema, test_needs_fk=True)
    Table('dingalings', metadata, Column('dingaling_id', sa.Integer, primary_key=True), Column('address_id', sa.Integer, ForeignKey('%semail_addresses.address_id' % schema_prefix, name='zz_email_add_id_fg', comment='di fk comment')), Column('id_user', sa.Integer, ForeignKey('%susers.user_id' % schema_prefix)), Column('data', sa.String(30), unique=True), sa.CheckConstraint('address_id > 0 AND address_id < 1000', name='address_id_gt_zero'), sa.UniqueConstraint('address_id', 'dingaling_id', name='zz_dingalings_multiple', comment='di unique comment'), schema=schema, test_needs_fk=True)
    Table('email_addresses', metadata, Column('address_id', sa.Integer), Column('remote_user_id', sa.Integer, ForeignKey(users.c.user_id)), Column('email_address', sa.String(20), index=True), sa.PrimaryKeyConstraint('address_id', name='email_ad_pk', comment='ea pk comment'), schema=schema, test_needs_fk=True)
    Table('comment_test', metadata, Column('id', sa.Integer, primary_key=True, comment='id comment'), Column('data', sa.String(20), comment='data % comment'), Column('d2', sa.String(20), comment='Comment types type speedily \' " \\ \'\' Fun!'), Column('d3', sa.String(42), comment='Comment\nwith\rescapes'), schema=schema, comment='the test % \' " \\ table comment')
    Table('no_constraints', metadata, Column('data', sa.String(20)), schema=schema, comment='no\nconstraints\rhas\x0cescaped\x0bcomment')
    if testing.requires.cross_schema_fk_reflection.enabled:
        if schema is None:
            Table('local_table', metadata, Column('id', sa.Integer, primary_key=True), Column('data', sa.String(20)), Column('remote_id', ForeignKey('%s.remote_table_2.id' % testing.config.test_schema)), test_needs_fk=True, schema=config.db.dialect.default_schema_name)
        else:
            Table('remote_table', metadata, Column('id', sa.Integer, primary_key=True), Column('local_id', ForeignKey('%s.local_table.id' % config.db.dialect.default_schema_name)), Column('data', sa.String(20)), schema=schema, test_needs_fk=True)
            Table('remote_table_2', metadata, Column('id', sa.Integer, primary_key=True), Column('data', sa.String(20)), schema=schema, test_needs_fk=True)
    if testing.requires.index_reflection.enabled:
        Index('users_t_idx', users.c.test1, users.c.test2, unique=True)
        Index('users_all_idx', users.c.user_id, users.c.test2, users.c.test1)
        if not schema:
            noncol_idx_test_nopk = Table('noncol_idx_test_nopk', metadata, Column('q', sa.String(5)), test_needs_fk=True)
            noncol_idx_test_pk = Table('noncol_idx_test_pk', metadata, Column('id', sa.Integer, primary_key=True), Column('q', sa.String(5)), test_needs_fk=True)
            if testing.requires.indexes_with_ascdesc.enabled and testing.requires.reflect_indexes_with_ascdesc.enabled:
                Index('noncol_idx_nopk', noncol_idx_test_nopk.c.q.desc())
                Index('noncol_idx_pk', noncol_idx_test_pk.c.q.desc())
    if testing.requires.view_column_reflection.enabled:
        cls.define_views(metadata, schema)
    if not schema and testing.requires.temp_table_reflection.enabled:
        cls.define_temp_tables(metadata)