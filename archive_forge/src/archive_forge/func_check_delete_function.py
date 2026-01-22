import pytest
import os
from ase.db import connect
def check_delete_function(db):
    db_size_full = os.path.getsize(db_name)
    db.delete([row.id for row in db.select()])
    db_size_empty = os.path.getsize(db_name)
    assert db_size_full > db_size_empty