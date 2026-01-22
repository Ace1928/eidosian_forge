from base64 import b64encode
from collections import namedtuple
import copy
import dataclasses
from functools import lru_cache
from io import BytesIO
import json
import logging
from numbers import Number
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
from typing import Union
import matplotlib as mpl
from matplotlib import _api, _afm, cbook, ft2font
from matplotlib._fontconfig_pattern import (
from matplotlib.rcsetup import _validators
def set_stretch(self, stretch):
    """
        Set the font stretch or width.

        Parameters
        ----------
        stretch : int or {'ultra-condensed', 'extra-condensed', 'condensed', 'semi-condensed', 'normal', 'semi-expanded', 'expanded', 'extra-expanded', 'ultra-expanded'}, default: :rc:`font.stretch`
            If int, must be in the range  0-1000.
        """
    if stretch is None:
        stretch = mpl.rcParams['font.stretch']
    if stretch in stretch_dict:
        self._stretch = stretch
        return
    try:
        stretch = int(stretch)
    except ValueError:
        pass
    else:
        if 0 <= stretch <= 1000:
            self._stretch = stretch
            return
    raise ValueError(f'stretch={stretch!r} is invalid')