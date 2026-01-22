import pytest
import os
from ase.db import connect
def test_delete_vacuum_context():
    with connect(db_name) as db:
        write_entries_to_db(db)
    with connect(db_name) as db:
        check_delete_function(db)