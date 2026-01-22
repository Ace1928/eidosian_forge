import pytest
import os
from ase.db import connect
def check_update_function(db):
    db_size_update = os.path.getsize(db_name)
    db.vacuum()
    db_size_update_vacuum = os.path.getsize(db_name)
    assert db_size_update > db_size_update_vacuum