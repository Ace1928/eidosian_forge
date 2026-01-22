import filecmp
import os
from pathlib import Path
import shutil
import sys
from matplotlib.testing import subprocess_run_for_testing
import pytest
def plot_file(num, suff=''):
    return img_dir / f'some_plots-{num}{suff}.png'