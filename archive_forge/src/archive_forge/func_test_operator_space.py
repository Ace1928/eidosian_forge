from __future__ import annotations
import io
from pathlib import Path
import platform
import re
import shlex
from xml.etree import ElementTree as ET
from typing import Any
import numpy as np
from packaging.version import parse as parse_version
import pyparsing
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import check_figures_equal, image_comparison
import matplotlib.pyplot as plt
from matplotlib import mathtext, _mathtext
@check_figures_equal(extensions=['png'])
def test_operator_space(fig_test, fig_ref):
    fig_test.text(0.1, 0.1, '$\\log 6$')
    fig_test.text(0.1, 0.2, '$\\log(6)$')
    fig_test.text(0.1, 0.3, '$\\arcsin 6$')
    fig_test.text(0.1, 0.4, '$\\arcsin|6|$')
    fig_test.text(0.1, 0.5, '$\\operatorname{op} 6$')
    fig_test.text(0.1, 0.6, '$\\operatorname{op}[6]$')
    fig_test.text(0.1, 0.7, '$\\cos^2$')
    fig_test.text(0.1, 0.8, '$\\log_2$')
    fig_test.text(0.1, 0.9, '$\\sin^2 \\cos$')
    fig_ref.text(0.1, 0.1, '$\\mathrm{log\\,}6$')
    fig_ref.text(0.1, 0.2, '$\\mathrm{log}(6)$')
    fig_ref.text(0.1, 0.3, '$\\mathrm{arcsin\\,}6$')
    fig_ref.text(0.1, 0.4, '$\\mathrm{arcsin}|6|$')
    fig_ref.text(0.1, 0.5, '$\\mathrm{op\\,}6$')
    fig_ref.text(0.1, 0.6, '$\\mathrm{op}[6]$')
    fig_ref.text(0.1, 0.7, '$\\mathrm{cos}^2$')
    fig_ref.text(0.1, 0.8, '$\\mathrm{log}_2$')
    fig_ref.text(0.1, 0.9, '$\\mathrm{sin}^2 \\mathrm{\\,cos}$')