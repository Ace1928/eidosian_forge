import ase.db
from ase.io.formats import string2index
def read_db(filename, index, **kwargs):
    db = ase.db.connect(filename, serial=True, **kwargs)
    if isinstance(index, str):
        try:
            index = string2index(index)
        except ValueError:
            pass
    if isinstance(index, int):
        index = slice(index, index + 1 or None)
    if isinstance(index, str):
        for row in db.select(index):
            yield row.toatoms()
    else:
        start, stop, step = index.indices(db.count())
        if start == stop:
            return
        assert step == 1
        for row in db.select(offset=start, limit=stop - start):
            yield row.toatoms()