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
def test_has_sequence_cache(self, connection, metadata):
    insp = inspect(connection)
    eq_(insp.has_sequence('user_id_seq'), True)
    ss = normalize_sequence(config, Sequence('new_seq', metadata=metadata))
    eq_(insp.has_sequence('new_seq'), False)
    ss.create(connection)
    try:
        eq_(insp.has_sequence('new_seq'), False)
        insp.clear_cache()
        eq_(insp.has_sequence('new_seq'), True)
    finally:
        ss.drop(connection)