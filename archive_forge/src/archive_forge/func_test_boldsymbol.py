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
def test_boldsymbol(fig_test, fig_ref):
    fig_test.text(0.1, 0.2, '$\\boldsymbol{\\mathrm{abc0123\\alpha}}$')
    fig_ref.text(0.1, 0.2, '$\\mathrm{abc0123\\alpha}$')