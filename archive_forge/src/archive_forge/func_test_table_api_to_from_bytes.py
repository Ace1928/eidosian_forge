import pytest
from spacy.lookups import Lookups, Table
from spacy.strings import get_string_id
from spacy.vocab import Vocab
from ..util import make_tempdir
def test_table_api_to_from_bytes():
    data = {'foo': 'bar', 'hello': 'world', 'abc': 123}
    table = Table(name='table', data=data)
    table_bytes = table.to_bytes()
    new_table = Table().from_bytes(table_bytes)
    assert new_table.name == 'table'
    assert len(new_table) == 3
    assert new_table['foo'] == 'bar'
    assert new_table[get_string_id('foo')] == 'bar'
    new_table2 = Table(data={'def': 456})
    new_table2.from_bytes(table_bytes)
    assert len(new_table2) == 3
    assert 'def' not in new_table2