import datetime
from io import BytesIO
import os
import shutil
import numpy as np
from packaging.version import parse as parse_version
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.testing import _has_tex_package, _check_for_pgf
from matplotlib.testing.exceptions import ImageComparisonFailure
from matplotlib.testing.compare import compare_images
from matplotlib.backends.backend_pgf import PdfPages
from matplotlib.testing.decorators import (
from matplotlib.testing._markers import (
@needs_pgf_xelatex
def test_multipage_keep_empty(tmp_path):
    os.chdir(tmp_path)
    with pytest.warns(mpl.MatplotlibDeprecationWarning), PdfPages('a.pdf') as pdf:
        pass
    assert os.path.exists('a.pdf')
    with pytest.warns(mpl.MatplotlibDeprecationWarning), PdfPages('b.pdf', keep_empty=True) as pdf:
        pass
    assert os.path.exists('b.pdf')
    with PdfPages('c.pdf', keep_empty=False) as pdf:
        pass
    assert not os.path.exists('c.pdf')
    with PdfPages('d.pdf') as pdf:
        pdf.savefig(plt.figure())
    assert os.path.exists('d.pdf')
    with pytest.warns(mpl.MatplotlibDeprecationWarning), PdfPages('e.pdf', keep_empty=True) as pdf:
        pdf.savefig(plt.figure())
    assert os.path.exists('e.pdf')
    with PdfPages('f.pdf', keep_empty=False) as pdf:
        pdf.savefig(plt.figure())
    assert os.path.exists('f.pdf')