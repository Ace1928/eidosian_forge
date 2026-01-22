import pytest
from ase import Atoms
from ase.db import connect
from ase.db.web import Session
def handle_query(args) -> str:
    """Converts request args to ase.db query string."""
    return args['query']