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
@pytest.mark.parametrize('index, text', enumerate(svgastext_math_tests), ids=range(len(svgastext_math_tests)))
@pytest.mark.parametrize('fontset', ['cm', 'dejavusans'])
@pytest.mark.parametrize('baseline_images', ['mathtext0'], indirect=True)
@image_comparison(baseline_images=None, extensions=['svg'], savefig_kwarg={'metadata': {'Creator': None, 'Date': None, 'Format': None, 'Type': None}})
def test_mathtext_rendering_svgastext(baseline_images, fontset, index, text):
    mpl.rcParams['mathtext.fontset'] = fontset
    mpl.rcParams['svg.fonttype'] = 'none'
    fig = plt.figure(figsize=(5.25, 0.75))
    fig.patch.set(visible=False)
    fig.text(0.5, 0.5, text, horizontalalignment='center', verticalalignment='center')