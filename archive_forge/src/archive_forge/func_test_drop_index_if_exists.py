import random
from . import testing
from .. import config
from .. import fixtures
from .. import util
from ..assertions import eq_
from ..assertions import is_false
from ..assertions import is_true
from ..config import requirements
from ..schema import Table
from ... import CheckConstraint
from ... import Column
from ... import ForeignKeyConstraint
from ... import Index
from ... import inspect
from ... import Integer
from ... import schema
from ... import String
from ... import UniqueConstraint
@requirements.index_ddl_if_exists
@util.provide_metadata
def test_drop_index_if_exists(self, connection):
    table, idx = self._table_index_fixture()
    table.create(connection)
    is_true('test_index' in [ix['name'] for ix in inspect(connection).get_indexes('test_table')])
    connection.execute(schema.DropIndex(idx, if_exists=True))
    is_false('test_index' in [ix['name'] for ix in inspect(connection).get_indexes('test_table')])
    connection.execute(schema.DropIndex(idx, if_exists=True))