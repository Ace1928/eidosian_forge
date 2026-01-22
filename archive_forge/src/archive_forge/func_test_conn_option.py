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
def test_conn_option(self):
    engine = self._fixture(False)
    with engine.connect() as conn:
        result = conn.execution_options(stream_results=True).exec_driver_sql('select 1')
        assert self._is_server_side(result.cursor)
        result.close()