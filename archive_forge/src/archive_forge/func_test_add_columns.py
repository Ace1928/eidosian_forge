import pytest
from ase import Atoms
from ase.db import connect
from ase.db.web import Session
def test_add_columns(database):
    """Test that all keys can be added also for row withous keys."""
    pytest.importorskip('flask')
    session = Session('name')
    project = {'default_columns': ['bar'], 'handle_query_function': handle_query}
    session.update('query', '', {'query': 'id=2'}, project)
    table = session.create_table(database, 'id', ['foo'])
    assert table.columns == ['bar']
    assert 'foo' in table.addcolumns