import pytest
from spacy.lookups import Lookups, Table
from spacy.strings import get_string_id
from spacy.vocab import Vocab
from ..util import make_tempdir
def test_table_api():
    table = Table(name='table')
    assert table.name == 'table'
    assert len(table) == 0
    assert 'abc' not in table
    data = {'foo': 'bar', 'hello': 'world'}
    table = Table(name='table', data=data)
    assert len(table) == len(data)
    assert 'foo' in table
    assert get_string_id('foo') in table
    assert table['foo'] == 'bar'
    assert table[get_string_id('foo')] == 'bar'
    assert table.get('foo') == 'bar'
    assert table.get('abc') is None
    table['abc'] = 123
    assert table['abc'] == 123
    assert table[get_string_id('abc')] == 123
    table.set('def', 456)
    assert table['def'] == 456
    assert table[get_string_id('def')] == 456