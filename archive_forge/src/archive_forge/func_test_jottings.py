import os.path
import io
from scipy.io.matlab._mio5 import MatFile5Reader
def test_jottings():
    fname = os.path.join(test_data_path, 'parabola.mat')
    read_workspace_vars(fname)