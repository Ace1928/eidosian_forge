from os import path
import pickle
import numpy as np
def test_unpickle_numpy_2_0_file(self):
    """
        Checks that NumPy 1.26 and pickle is able to load pickles
        created with NumPy 2.0 without errors/warnings.
        """
    with open(self.filename, mode='rb') as file:
        content = file.read()
    assert b'numpy._core.multiarray' in content
    arr = pickle.loads(content, encoding='latin1')
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (73,) and arr.dtype == np.float64