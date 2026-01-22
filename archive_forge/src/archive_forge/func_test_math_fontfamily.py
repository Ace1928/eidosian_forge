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
@image_comparison(baseline_images=['math_fontfamily_image.png'], savefig_kwarg={'dpi': 40})
def test_math_fontfamily():
    fig = plt.figure(figsize=(10, 3))
    fig.text(0.2, 0.7, '$This\\ text\\ should\\ have\\ one\\ font$', size=24, math_fontfamily='dejavusans')
    fig.text(0.2, 0.3, '$This\\ text\\ should\\ have\\ another$', size=24, math_fontfamily='stix')