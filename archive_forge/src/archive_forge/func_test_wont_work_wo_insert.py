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
def test_wont_work_wo_insert(self):
    default_schema_name = config.db.dialect.default_schema_name
    eng = engines.testing_engine()

    @event.listens_for(eng, 'connect')
    def on_connect(dbapi_connection, connection_record):
        set_default_schema_on_connection(config, dbapi_connection, config.test_schema)
    with eng.connect() as conn:
        what_it_should_be = eng.dialect._get_default_schema_name(conn)
        eq_(what_it_should_be, config.test_schema)
    eq_(eng.dialect.default_schema_name, default_schema_name)