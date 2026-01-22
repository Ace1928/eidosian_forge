import pytest
import os
from ase.db import connect
def test_update_vacuum():
    db = connect(db_name)
    write_entries_to_db(db)
    update_keys_in_db(db)
    check_update_function(db)