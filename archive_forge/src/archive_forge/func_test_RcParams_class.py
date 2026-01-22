import copy
import os
from pathlib import Path
import subprocess
import sys
from unittest import mock
from cycler import cycler, Cycler
import pytest
import matplotlib as mpl
from matplotlib import _api, _c_internal_utils
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.rcsetup import (
def test_RcParams_class():
    rc = mpl.RcParams({'font.cursive': ['Apple Chancery', 'Textile', 'Zapf Chancery', 'cursive'], 'font.family': 'sans-serif', 'font.weight': 'normal', 'font.size': 12})
    expected_repr = "\nRcParams({'font.cursive': ['Apple Chancery',\n                           'Textile',\n                           'Zapf Chancery',\n                           'cursive'],\n          'font.family': ['sans-serif'],\n          'font.size': 12.0,\n          'font.weight': 'normal'})".lstrip()
    assert expected_repr == repr(rc)
    expected_str = "\nfont.cursive: ['Apple Chancery', 'Textile', 'Zapf Chancery', 'cursive']\nfont.family: ['sans-serif']\nfont.size: 12.0\nfont.weight: normal".lstrip()
    assert expected_str == str(rc)
    assert ['font.cursive', 'font.size'] == sorted(rc.find_all('i[vz]'))
    assert ['font.family'] == list(rc.find_all('family'))