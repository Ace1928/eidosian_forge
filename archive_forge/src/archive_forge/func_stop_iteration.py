from typing import Dict, Optional, Tuple
import torch
from torch import Tensor
from . import _linalg_utils as _utils
from .overrides import handle_torch_function, has_torch_function
def stop_iteration(self):
    """Return True to stop iterations.

        Note that tracker (if defined) can force-stop iterations by
        setting ``worker.bvars['force_stop'] = True``.
        """
    return self.bvars.get('force_stop', False) or self.ivars['iterations_left'] == 0 or self.ivars['converged_count'] >= self.iparams['k']