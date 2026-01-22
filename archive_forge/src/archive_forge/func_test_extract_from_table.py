import os
import pytest
from ase.db import connect
from ase import Atoms
import numpy as np
def test_extract_from_table(db_name):
    atoms = Atoms()
    db = connect(db_name)
    uid = db.write(atoms, external_tables={'insert_tab': {'rate': 12.0, 'rate1': -10.0}})
    row = db.get(id=uid)
    assert abs(row['insert_tab']['rate'] - 12.0) < 1e-08
    assert abs(row['insert_tab']['rate1'] + 10.0) < 1e-08