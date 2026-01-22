from collections import OrderedDict
from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from cmdstanpy.cmdstan_args import Method, OptimizeArgs
from cmdstanpy.utils import get_logger, scan_optimize_csv
from .metadata import InferenceMetadata
from .runset import RunSet
@property
def optimized_iterations_np(self) -> Optional[np.ndarray]:
    """
        Returns all saved iterations from the optimizer and final estimate
        as a numpy.ndarray which contains all optimizer outputs, i.e.,
        the value for `lp__` as well as all Stan program variables.

        """
    if not self._save_iterations:
        get_logger().warning('Intermediate iterations not saved to CSV output file. Rerun the optimize method with "save_iterations=True".')
        return None
    if not self.converged:
        get_logger().warning('Invalid estimate, optimization failed to converge.')
    return self._all_iters