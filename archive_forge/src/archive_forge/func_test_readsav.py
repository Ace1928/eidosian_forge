from pathlib import Path
import numpy as np
import scipy.io
import scipy.io.wavfile
from scipy._lib._tmpdirs import tempdir
import scipy.sparse
def test_readsav(self):
    path = Path(__file__).parent / 'data/scalar_string.sav'
    scipy.io.readsav(path)