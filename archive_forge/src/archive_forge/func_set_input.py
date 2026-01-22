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
def set_input(self, parameter, val):
    """
        Set interface input value or nodewrapper attribute
        Priority goes to interface.
        """
    logger.debug('setting nodelevel(%s) input %s = %s', str(self), parameter, str(val))
    self._set_mapnode_input(parameter, deepcopy(val))