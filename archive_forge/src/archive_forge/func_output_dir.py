from collections import OrderedDict, defaultdict
import os
import os.path as op
from pathlib import Path
import shutil
import socket
from copy import deepcopy
from glob import glob
from logging import INFO
from tempfile import mkdtemp
from ... import config, logging
from ...utils.misc import flatten, unflatten, str2bool, dict_diff
from ...utils.filemanip import (
from ...interfaces.base import (
from ...interfaces.base.specs import get_filecopy_info
from .utils import (
from .base import EngineBase
def output_dir(self):
    """Return the location of the output directory for the node"""
    if self._output_dir:
        return self._output_dir
    if self.base_dir is None:
        self.base_dir = mkdtemp()
    outputdir = self.base_dir
    if self._hierarchy:
        outputdir = op.join(outputdir, *self._hierarchy.split('.'))
    if self.parameterization:
        maxlen = 252 if str2bool(self.config['execution']['parameterize_dirs']) else 32
        params_str = [_parameterization_dir(str(p), maxlen) for p in self.parameterization]
        outputdir = op.join(outputdir, *params_str)
    self._output_dir = op.realpath(op.join(outputdir, self.name))
    return self._output_dir