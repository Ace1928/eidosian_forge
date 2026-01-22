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
def hash_exists(self, updatehash=False):
    """
        Decorate the new `is_cached` method with hash updating
        to maintain backwards compatibility.
        """
    cached, updated = self.is_cached(rm_outdated=True)
    outdir = self.output_dir()
    hashfile = op.join(outdir, '_0x%s.json' % self._hashvalue)
    if updated:
        return (True, self._hashvalue, hashfile, self._hashed_inputs)
    if cached and updatehash:
        logger.debug('[Node] Updating hash: %s', self._hashvalue)
        _save_hashfile(hashfile, self._hashed_inputs)
    return (cached, self._hashvalue, hashfile, self._hashed_inputs)