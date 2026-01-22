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
def test_all_levels(self):
    levels = requirements.get_isolation_levels(config)
    all_levels = levels['supported']
    for level in set(all_levels).difference(['AUTOCOMMIT']):
        with config.db.connect() as conn:
            conn.execution_options(isolation_level=level)
            eq_(conn.get_isolation_level(), level)
            trans = conn.begin()
            trans.rollback()
            eq_(conn.get_isolation_level(), level)
        with config.db.connect() as conn:
            eq_(conn.get_isolation_level(), levels['default'])