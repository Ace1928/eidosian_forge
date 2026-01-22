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
def new_saved_frame_seq(self):
    if self._save_seq:
        self._old_saved_seq = list(self._save_seq)
        return iter(self._old_saved_seq)
    elif self._save_count is None:
        frame_seq = self.new_frame_seq()

        def gen():
            try:
                while True:
                    yield next(frame_seq)
            except StopIteration:
                pass
        return gen()
    else:
        return itertools.islice(self.new_frame_seq(), self._save_count)