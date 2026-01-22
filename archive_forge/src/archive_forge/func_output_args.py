import abc
import base64
import contextlib
from io import BytesIO, TextIOWrapper
import itertools
import logging
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import uuid
import warnings
import numpy as np
from PIL import Image
import matplotlib as mpl
from matplotlib._animation_data import (
from matplotlib import _api, cbook
import matplotlib.colors as mcolors
@property
def output_args(self):
    args = []
    if Path(self.outfile).suffix == '.gif':
        self.codec = 'gif'
    else:
        args.extend(['-vcodec', self.codec])
    extra_args = self.extra_args if self.extra_args is not None else mpl.rcParams[self._args_key]
    if self.codec == 'h264' and '-pix_fmt' not in extra_args:
        args.extend(['-pix_fmt', 'yuv420p'])
    elif self.codec == 'gif' and '-filter_complex' not in extra_args:
        args.extend(['-filter_complex', 'split [a][b];[a] palettegen [p];[b][p] paletteuse'])
    if self.bitrate > 0:
        args.extend(['-b', '%dk' % self.bitrate])
    for k, v in self.metadata.items():
        args.extend(['-metadata', f'{k}={v}'])
    args.extend(extra_args)
    return args + ['-y', self.outfile]