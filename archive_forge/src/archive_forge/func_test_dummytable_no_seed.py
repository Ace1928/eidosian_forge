import random as pyrandom
import time
from functools import partial
from petl.util.random import randomseed, randomtable, RandomTable, dummytable, DummyTable
def test_dummytable_no_seed():
    """
    Ensure that dummytable provides a table with the right number of rows
    and columns when not provided with a seed.
    """
    rows = 35
    table = dummytable(numrows=rows)
    assert len(table[0]) == 3
    assert len(table) == rows + 1