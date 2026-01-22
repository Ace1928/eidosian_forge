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
@needs_ghostscript
@pytest.mark.backend('pgf')
def test_tex_special_chars(tmp_path):
    fig = plt.figure()
    fig.text(0.5, 0.5, '%_^ $a_b^c$')
    buf = BytesIO()
    fig.savefig(buf, format='png', backend='pgf')
    buf.seek(0)
    t = plt.imread(buf)
    assert not (t == 1).all()