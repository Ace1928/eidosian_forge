import pytest
from ase import Atoms
from ase.db import connect
from ase.db.web import Session
def test_favicon(client):
    assert client.get('/favicon.ico').status_code == 308
    assert client.get('/favicon.ico/').status_code == 204