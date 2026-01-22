import functools
import importlib
import os
import platform
import subprocess
import sys
import pytest
from matplotlib import _c_internal_utils
from matplotlib.testing import subprocess_run_helper
@_isolated_tk_test(success_count=6)
def test_blit():
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.backends.backend_tkagg
    from matplotlib.backends import _backend_tk, _tkagg
    fig, ax = plt.subplots()
    photoimage = fig.canvas._tkphoto
    data = np.ones((4, 4, 4))
    height, width = data.shape[:2]
    dataptr = (height, width, data.ctypes.data)
    bad_boxes = ((-1, 2, 0, 2), (2, 0, 0, 2), (1, 6, 0, 2), (0, 2, -1, 2), (0, 2, 2, 0), (0, 2, 1, 6))
    for bad_box in bad_boxes:
        try:
            _tkagg.blit(photoimage.tk.interpaddr(), str(photoimage), dataptr, 0, (0, 1, 2, 3), bad_box)
        except ValueError:
            print('success')
    plt.close(fig)
    _backend_tk.blit(photoimage, data, (0, 1, 2, 3))