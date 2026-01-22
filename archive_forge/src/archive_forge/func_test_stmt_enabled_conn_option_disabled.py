import datetime
from .. import engines
from .. import fixtures
from ..assertions import eq_
from ..config import requirements
from ..schema import Column
from ..schema import Table
from ... import DateTime
from ... import func
from ... import Integer
from ... import select
from ... import sql
from ... import String
from ... import testing
from ... import text
def test_stmt_enabled_conn_option_disabled(self):
    engine = self._fixture(False)
    s = select(1).execution_options(stream_results=True)
    with engine.connect() as conn:
        result = conn.execution_options(stream_results=False).execute(s)
        assert not self._is_server_side(result.cursor)