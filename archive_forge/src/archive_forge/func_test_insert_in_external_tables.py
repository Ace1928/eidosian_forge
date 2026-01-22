import os
import pytest
from ase.db import connect
from ase import Atoms
import numpy as np
def test_insert_in_external_tables(db_name):
    atoms = Atoms()
    db = connect(db_name)
    uid = db.write(atoms, external_tables={'insert_tab': {'rate': 1.0, 'rate1': -2.0}})
    db.delete([uid])
    con = db._connect()
    cur = con.cursor()
    sql = 'SELECT * FROM insert_tab WHERE ID=?'
    cur.execute(sql, (uid,))
    entries = [x for x in cur.fetchall()]
    if db.connection is None:
        con.close()
    assert not entries
    with pytest.raises(ValueError):
        db.write(atoms, external_tables={'insert_tab': {'rate': 'something'}})
    db.write(atoms, external_tables={'insert_tab': {'rate': np.float32(1.0)}})
    db.write(atoms, external_tables={'insert_tab': {'rate': np.float64(1.0)}})
    with pytest.raises(ValueError):
        db.write(atoms, external_tables={'insert_tab': {'rate': np.int32(1.0)}})
    with pytest.raises(ValueError):
        db.write(atoms, external_tables={'insert_tab': {'rate': np.int64(1.0)}})
    db.write(atoms, external_tables={'integer_tab': {'rate': 1}})
    db.write(atoms, external_tables={'integer_tab': {'rate': np.int32(1)}})
    db.write(atoms, external_tables={'integer_tab': {'rate': np.int64(1)}})
    with pytest.raises(ValueError):
        db.write(atoms, external_tables={'integer_tab': {'rate': np.float32(1)}})
    with pytest.raises(ValueError):
        db.write(atoms, external_tables={'integer_tab': {'rate': np.float64(1)}})
    with pytest.raises(ValueError):
        db.write(atoms, external_tables={'integer_tab': {'rate': 1, 'rate2': 2.0}})
    from ase.db.sqlite import all_tables
    for tab_name in all_tables:
        with pytest.raises(ValueError):
            db.write(atoms, external_tables={tab_name: {'value': 1}})