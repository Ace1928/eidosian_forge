import pytest
import os
from ase.db import connect
def write_entries_to_db(db, n_entries_db=30):
    for i in range(n_entries_db):
        db.reserve(mykey=f'test_{i}')