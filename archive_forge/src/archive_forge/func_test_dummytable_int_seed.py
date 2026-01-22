import random as pyrandom
import time
from functools import partial
from petl.util.random import randomseed, randomtable, RandomTable, dummytable, DummyTable
def test_dummytable_int_seed():
    """
    Ensure that dummytable provides a table with the right number of rows
    and columns when provided with an integer as a seed.
    """
    rows = 35
    seed = 42
    table = dummytable(numrows=rows, seed=seed)
    assert len(table[0]) == 3
    assert len(table) == rows + 1