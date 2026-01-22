import os
import pytest
from ase.db import connect
from ase import Atoms
import numpy as np
def test_external_table_upon_update(db_name):
    db = connect(db_name)
    no_features = 500
    ext_table = dict(((i, i) for i in range(no_features)))
    atoms = Atoms('Pb', positions=[[0, 0, 0]])
    uid = db.write(atoms)
    db.update(uid, external_tables={'sys': ext_table})