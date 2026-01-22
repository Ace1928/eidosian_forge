import pytest
from spacy.lookups import Lookups, Table
from spacy.strings import get_string_id
from spacy.vocab import Vocab
from ..util import make_tempdir
def test_lookups_api():
    table_name = 'test'
    data = {'foo': 'bar', 'hello': 'world'}
    lookups = Lookups()
    lookups.add_table(table_name, data)
    assert len(lookups) == 1
    assert table_name in lookups
    assert lookups.has_table(table_name)
    table = lookups.get_table(table_name)
    assert table.name == table_name
    assert len(table) == 2
    assert table['hello'] == 'world'
    table['a'] = 'b'
    assert table['a'] == 'b'
    table = lookups.get_table(table_name)
    assert len(table) == 3
    with pytest.raises(KeyError):
        lookups.get_table('xyz')
    with pytest.raises(ValueError):
        lookups.add_table(table_name)
    table = lookups.remove_table(table_name)
    assert table.name == table_name
    assert len(lookups) == 0
    assert table_name not in lookups
    with pytest.raises(KeyError):
        lookups.get_table(table_name)