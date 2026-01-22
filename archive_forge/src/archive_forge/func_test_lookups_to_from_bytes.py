import pytest
from spacy.lookups import Lookups, Table
from spacy.strings import get_string_id
from spacy.vocab import Vocab
from ..util import make_tempdir
def test_lookups_to_from_bytes():
    lookups = Lookups()
    lookups.add_table('table1', {'foo': 'bar', 'hello': 'world'})
    lookups.add_table('table2', {'a': 1, 'b': 2, 'c': 3})
    lookups_bytes = lookups.to_bytes()
    new_lookups = Lookups()
    new_lookups.from_bytes(lookups_bytes)
    assert len(new_lookups) == 2
    assert 'table1' in new_lookups
    assert 'table2' in new_lookups
    table1 = new_lookups.get_table('table1')
    assert len(table1) == 2
    assert table1['foo'] == 'bar'
    table2 = new_lookups.get_table('table2')
    assert len(table2) == 3
    assert table2['b'] == 2
    assert new_lookups.to_bytes() == lookups_bytes