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
def score_stretch(self, stretch1, stretch2):
    """
        Return a match score between *stretch1* and *stretch2*.

        The result is the absolute value of the difference between the
        CSS numeric values of *stretch1* and *stretch2*, normalized
        between 0.0 and 1.0.
        """
    try:
        stretchval1 = int(stretch1)
    except ValueError:
        stretchval1 = stretch_dict.get(stretch1, 500)
    try:
        stretchval2 = int(stretch2)
    except ValueError:
        stretchval2 = stretch_dict.get(stretch2, 500)
    return abs(stretchval1 - stretchval2) / 1000.0