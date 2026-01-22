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
def test_proxy_transaction_rollback(self):
    context = self._fixture({'transaction_per_migration': True, 'transactional_ddl': True})
    is_false(self.conn.in_transaction())
    proxy = context.begin_transaction(_per_migration=True)
    is_true(self.conn.in_transaction())
    proxy.rollback()
    is_false(self.conn.in_transaction())