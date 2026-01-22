import os
import pytest
from ase.db import connect
from ase import Atoms
from ase.calculators.emt import EMT
from ase.build import molecule
def test_read_write_bool_key_value_pair(db, h2o):
    uid = db.write(h2o, is_water=True, is_solid=False)
    row = db.get(id=uid)
    assert row.is_water
    assert not row.is_solid