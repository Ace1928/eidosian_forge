import importlib
from . import testing
from .. import assert_raises
from .. import config
from .. import engines
from .. import eq_
from .. import fixtures
from .. import is_not_none
from .. import is_true
from .. import ne_
from .. import provide_metadata
from ..assertions import expect_raises
from ..assertions import expect_raises_message
from ..config import requirements
from ..provision import set_default_schema_on_connection
from ..schema import Column
from ..schema import Table
from ... import bindparam
from ... import dialects
from ... import event
from ... import exc
from ... import Integer
from ... import literal_column
from ... import select
from ... import String
from ...sql.compiler import Compiled
from ...util import inspect_getfullargspec
def test_non_default_isolation_level(self):
    non_default = self._get_non_default_isolation_level()
    with config.db.connect() as conn:
        existing = conn.get_isolation_level()
        ne_(existing, non_default)
        conn.execution_options(isolation_level=non_default)
        eq_(conn.get_isolation_level(), non_default)
        conn.dialect.reset_isolation_level(conn.connection.dbapi_connection)
        eq_(conn.get_isolation_level(), existing)