from .. import config
from .. import fixtures
from ..assertions import eq_
from ..assertions import is_true
from ..config import requirements
from ..provision import normalize_sequence
from ..schema import Column
from ..schema import Table
from ... import inspect
from ... import Integer
from ... import MetaData
from ... import Sequence
from ... import String
from ... import testing
def test_insert_roundtrip_no_implicit_returning(self, connection):
    connection.execute(self.tables.seq_no_returning.insert(), dict(data='some data'))
    self._assert_round_trip(self.tables.seq_no_returning, connection)