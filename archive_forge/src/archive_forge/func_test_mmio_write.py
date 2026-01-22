from pathlib import Path
import numpy as np
import scipy.io
import scipy.io.wavfile
from scipy._lib._tmpdirs import tempdir
import scipy.sparse
def test_mmio_write(self):
    with tempdir() as temp_dir:
        data = scipy.sparse.csr_matrix(scipy.sparse.eye(3))
        path = Path(temp_dir) / 'data.mtx'
        scipy.io.mmwrite(path, data)