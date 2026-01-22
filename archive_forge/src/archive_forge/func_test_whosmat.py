from pathlib import Path
import numpy as np
import scipy.io
import scipy.io.wavfile
from scipy._lib._tmpdirs import tempdir
import scipy.sparse
def test_whosmat(self):
    with tempdir() as temp_dir:
        path = Path(temp_dir) / 'data.mat'
        scipy.io.savemat(str(path), {'data': self.data})
        contents = scipy.io.whosmat(path)
        assert contents[0] == ('data', (1, 5), 'int64')