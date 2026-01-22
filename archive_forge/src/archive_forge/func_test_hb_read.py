from pathlib import Path
import numpy as np
import scipy.io
import scipy.io.wavfile
from scipy._lib._tmpdirs import tempdir
import scipy.sparse
def test_hb_read(self):
    with tempdir() as temp_dir:
        data = scipy.sparse.csr_matrix(scipy.sparse.eye(3))
        path = Path(temp_dir) / 'data.hb'
        scipy.io.hb_write(str(path), data)
        data_new = scipy.io.hb_read(path)
        assert (data_new != data).nnz == 0