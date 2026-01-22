import os
from pathlib import Path
import re
import subprocess
import sys
import matplotlib.pyplot as plt
from matplotlib.texmanager import TexManager
from matplotlib.testing._markers import needs_usetex
import pytest
@needs_usetex
def test_unicode_characters():
    plt.rcParams['text.usetex'] = True
    fig, ax = plt.subplots()
    ax.set_ylabel('\\textit{Velocity (°/sec)}')
    ax.set_xlabel('¼Öøæ')
    fig.canvas.draw()
    with pytest.raises(RuntimeError):
        ax.set_title('☃')
        fig.canvas.draw()