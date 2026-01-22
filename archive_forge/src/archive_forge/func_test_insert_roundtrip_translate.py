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
@testing.combinations((True,), (False,), argnames='implicit_returning')
@testing.requires.schemas
def test_insert_roundtrip_translate(self, connection, implicit_returning):
    seq_no_returning = Table('seq_no_returning_sch', MetaData(), Column('id', Integer, normalize_sequence(config, Sequence('noret_sch_id_seq', schema='alt_schema')), primary_key=True), Column('data', String(50)), implicit_returning=implicit_returning, schema='alt_schema')
    connection = connection.execution_options(schema_translate_map={'alt_schema': config.test_schema})
    connection.execute(seq_no_returning.insert(), dict(data='some data'))
    self._assert_round_trip(seq_no_returning, connection)