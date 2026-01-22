import functools
import hashlib
import logging
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, dviread
@classmethod
def get_font_preamble(cls):
    """
        Return a string containing font configuration for the tex preamble.
        """
    font_preamble, command = cls._get_font_preamble_and_command()
    return font_preamble