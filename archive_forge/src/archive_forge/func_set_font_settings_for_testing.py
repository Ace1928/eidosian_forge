from pathlib import Path
from tempfile import TemporaryDirectory
import locale
import logging
import os
import subprocess
import sys
import matplotlib as mpl
from matplotlib import _api
def set_font_settings_for_testing():
    mpl.rcParams['font.family'] = 'DejaVu Sans'
    mpl.rcParams['text.hinting'] = 'none'
    mpl.rcParams['text.hinting_factor'] = 8