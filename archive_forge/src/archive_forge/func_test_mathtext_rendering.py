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
@pytest.mark.parametrize('index, text', enumerate(math_tests), ids=range(len(math_tests)))
@pytest.mark.parametrize('fontset', ['cm', 'stix', 'stixsans', 'dejavusans', 'dejavuserif'])
@pytest.mark.parametrize('baseline_images', ['mathtext'], indirect=True)
@image_comparison(baseline_images=None, tol=0.011 if platform.machine() in ('ppc64le', 's390x') else 0)
def test_mathtext_rendering(baseline_images, fontset, index, text):
    mpl.rcParams['mathtext.fontset'] = fontset
    fig = plt.figure(figsize=(5.25, 0.75))
    fig.text(0.5, 0.5, text, horizontalalignment='center', verticalalignment='center')