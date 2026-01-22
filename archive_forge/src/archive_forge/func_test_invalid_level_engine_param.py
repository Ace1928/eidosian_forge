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
@testing.requires.get_isolation_level_values
@testing.requires.dialect_level_isolation_level_param
def test_invalid_level_engine_param(self, testing_engine):
    """test for the new get_isolation_level_values() method
        and support for the dialect-level 'isolation_level' parameter.

        """
    eng = testing_engine(options=dict(isolation_level='FOO'))
    with expect_raises_message(exc.ArgumentError, "Invalid value '%s' for isolation_level. Valid isolation levels for '%s' are %s" % ('FOO', eng.dialect.name, ', '.join(requirements.get_isolation_levels(config)['supported']))):
        eng.connect()