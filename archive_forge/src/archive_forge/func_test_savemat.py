from pathlib import Path
import numpy as np
import scipy.io
import scipy.io.wavfile
from scipy._lib._tmpdirs import tempdir
import scipy.sparse
def test_savemat(self):
    with tempdir() as temp_dir:
        path = Path(temp_dir) / 'data.mat'
        scipy.io.savemat(path, {'data': self.data})
        assert path.is_file()