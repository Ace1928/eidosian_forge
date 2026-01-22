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
def test_short_long_accents(fig_test, fig_ref):
    acc_map = _mathtext.Parser._accent_map
    short_accs = [s for s in acc_map if len(s) == 1]
    corresponding_long_accs = []
    for s in short_accs:
        l, = [l for l in acc_map if len(l) > 1 and acc_map[l] == acc_map[s]]
        corresponding_long_accs.append(l)
    fig_test.text(0, 0.5, '$' + ''.join((f'\\{s}a' for s in short_accs)) + '$')
    fig_ref.text(0, 0.5, '$' + ''.join((f'\\{l} a' for l in corresponding_long_accs)) + '$')