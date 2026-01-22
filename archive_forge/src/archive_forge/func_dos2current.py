import numpy as np
from ase.io.jsonio import read_json, write_json
def dos2current(bias, dos):
    return 5000.0 * dos ** 2 * (1 if bias > 0 else -1)