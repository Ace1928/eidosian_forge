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
@needs_pgf_pdflatex
@mpl.style.context('default')
@pytest.mark.backend('pgf')
def test_rcupdate():
    rc_sets = [{'font.family': 'sans-serif', 'font.size': 30, 'figure.subplot.left': 0.2, 'lines.markersize': 10, 'pgf.rcfonts': False, 'pgf.texsystem': 'xelatex'}, {'font.family': 'monospace', 'font.size': 10, 'figure.subplot.left': 0.1, 'lines.markersize': 20, 'pgf.rcfonts': False, 'pgf.texsystem': 'pdflatex', 'pgf.preamble': '\\usepackage[utf8x]{inputenc}\\usepackage[T1]{fontenc}\\usepackage{sfmath}'}]
    tol = [0, 13.2] if _old_gs_version else [0, 0]
    for i, rc_set in enumerate(rc_sets):
        with mpl.rc_context(rc_set):
            for substring, pkg in [('sfmath', 'sfmath'), ('utf8x', 'ucs')]:
                if substring in mpl.rcParams['pgf.preamble'] and (not _has_tex_package(pkg)):
                    pytest.skip(f'needs {pkg}.sty')
            create_figure()
            compare_figure(f'pgf_rcupdate{i + 1}.pdf', tol=tol[i])