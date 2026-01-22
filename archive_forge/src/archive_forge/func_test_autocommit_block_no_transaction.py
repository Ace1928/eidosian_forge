import io
from ...migration import MigrationContext
from ...testing import assert_raises
from ...testing import config
from ...testing import eq_
from ...testing import is_
from ...testing import is_false
from ...testing import is_not_
from ...testing import is_true
from ...testing import ne_
from ...testing.fixtures import TestBase
@config.requirements.autocommit_isolation
def test_autocommit_block_no_transaction(self):
    context = self._fixture({'transaction_per_migration': True})
    is_false(self.conn.in_transaction())
    with context.autocommit_block():
        is_true(context.connection.in_transaction())
        if self.is_sqlalchemy_future:
            is_(context.connection, self.conn)
        else:
            is_not_(context.connection, self.conn)
            is_false(self.conn.in_transaction())
        eq_(context.connection._execution_options['isolation_level'], 'AUTOCOMMIT')
    ne_(context.connection._execution_options.get('isolation_level', None), 'AUTOCOMMIT')
    is_false(self.conn.in_transaction())