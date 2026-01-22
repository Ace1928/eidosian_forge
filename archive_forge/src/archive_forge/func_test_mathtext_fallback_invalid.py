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
def test_mathtext_fallback_invalid():
    for fallback in ['abc', '']:
        with pytest.raises(ValueError, match='not a valid fallback font name'):
            mpl.rcParams['mathtext.fallback'] = fallback