import filecmp
import os
from pathlib import Path
import shutil
import sys
from matplotlib.testing import subprocess_run_for_testing
import pytest
def plot_directive_file(num):
    return doctree_dir.parent / 'plot_directive' / f'some_plots-{num}.png'