from pathlib import Path
from tempfile import TemporaryDirectory
import locale
import logging
import os
import subprocess
import sys
import matplotlib as mpl
from matplotlib import _api
def set_reproducibility_for_testing():
    mpl.rcParams['svg.hashsalt'] = 'matplotlib'