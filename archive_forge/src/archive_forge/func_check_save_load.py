from io import StringIO
import tempfile
import numpy as np
from numpy.testing import assert_equal, \
from scipy.sparse import coo_matrix, csc_matrix, rand
from scipy.io import hb_read, hb_write
def check_save_load(self, value):
    with tempfile.NamedTemporaryFile(mode='w+t') as file:
        hb_write(file, value)
        file.file.seek(0)
        value_loaded = hb_read(file)
    assert_csc_almost_equal(value, value_loaded)