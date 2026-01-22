import pytest
from ase import Atoms
from ase.db import connect
from ase.db.web import Session
def test_db_web(client):
    import io
    from ase.db.web import Session
    from ase.io import read
    c = client
    page = c.get('/').data.decode()
    sid = Session.next_id - 1
    assert 'foo' in page
    for url in [f'/update/{sid}/query/bla/?query=id=1', '/default/row/1']:
        resp = c.get(url)
        assert resp.status_code == 200
    for type in ['json', 'xyz', 'cif']:
        url = f'atoms/default/1/{type}'
        resp = c.get(url)
        assert resp.status_code == 200
        atoms = read(io.StringIO(resp.data.decode()), format=type)
        print(atoms.numbers)
        assert (atoms.numbers == [1, 1, 8]).all()