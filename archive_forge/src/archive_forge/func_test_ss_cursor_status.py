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
@testing.combinations(('global_string', True, 'select 1', True), ('global_text', True, text('select 1'), True), ('global_expr', True, select(1), True), ('global_off_explicit', False, text('select 1'), False), ('stmt_option', False, select(1).execution_options(stream_results=True), True), ('stmt_option_disabled', True, select(1).execution_options(stream_results=False), False), ('for_update_expr', True, select(1).with_for_update(), True), ('for_update_string', True, 'SELECT 1 FOR UPDATE', True, testing.skip_if(['sqlite', 'mssql'])), ('text_no_ss', False, text('select 42'), False), ('text_ss_option', False, text('select 42').execution_options(stream_results=True), True), id_='iaaa', argnames='engine_ss_arg, statement, cursor_ss_status')
def test_ss_cursor_status(self, engine_ss_arg, statement, cursor_ss_status):
    engine = self._fixture(engine_ss_arg)
    with engine.begin() as conn:
        if isinstance(statement, str):
            result = conn.exec_driver_sql(statement)
        else:
            result = conn.execute(statement)
        eq_(self._is_server_side(result.cursor), cursor_ss_status)
        result.close()