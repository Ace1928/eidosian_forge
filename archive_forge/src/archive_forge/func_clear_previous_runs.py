import os
import hashlib
import pickle
import time
import shutil
import glob
from ..interfaces.base import BaseInterface
from ..pipeline.engine import Node
from ..pipeline.engine.utils import modify_paths
def clear_previous_runs(self, warn=True):
    """Remove all the cache that where not used in the latest run of
        the memory object: i.e. since the corresponding Python object
        was created.

        Parameters
        ==========
        warn: boolean, optional
            If true, echoes warning messages for all directory
            removed
        """
    base_dir = self.base_dir
    latest_runs = read_log(os.path.join(base_dir, 'log.current'))
    self._clear_all_but(latest_runs, warn=warn)