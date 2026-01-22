import os
from pathlib import Path
import re
import subprocess
import sys
import matplotlib.pyplot as plt
from matplotlib.texmanager import TexManager
from matplotlib.testing._markers import needs_usetex
import pytest
def test_fontconfig_preamble():
    """Test that the preamble is included in the source."""
    plt.rcParams['text.usetex'] = True
    src1 = TexManager()._get_tex_source('', fontsize=12)
    plt.rcParams['text.latex.preamble'] = '\\usepackage{txfonts}'
    src2 = TexManager()._get_tex_source('', fontsize=12)
    assert src1 != src2