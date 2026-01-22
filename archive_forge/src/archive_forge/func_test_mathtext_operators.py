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
def test_mathtext_operators():
    test_str = '\n    \\increment \\smallin \\notsmallowns\n    \\smallowns \\QED \\rightangle\n    \\smallintclockwise \\smallvarointclockwise\n    \\smallointctrcclockwise\n    \\ratio \\minuscolon \\dotsminusdots\n    \\sinewave \\simneqq \\nlesssim\n    \\ngtrsim \\nlessgtr \\ngtrless\n    \\cupleftarrow \\oequal \\rightassert\n    \\rightModels \\hermitmatrix \\barvee\n    \\measuredrightangle \\varlrtriangle\n    \\equalparallel \\npreccurlyeq \\nsucccurlyeq\n    \\nsqsubseteq \\nsqsupseteq \\sqsubsetneq\n    \\sqsupsetneq  \\disin \\varisins\n    \\isins \\isindot \\varisinobar\n    \\isinobar \\isinvb \\isinE\n    \\nisd \\varnis \\nis\n    \\varniobar \\niobar \\bagmember\n    \\triangle'.split()
    fig = plt.figure()
    for x, i in enumerate(test_str):
        fig.text(0.5, (x + 0.5) / len(test_str), '${%s}$' % i)
    fig.draw_without_rendering()