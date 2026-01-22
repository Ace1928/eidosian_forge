from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from cmdstanpy.cmdstan_args import Method
from cmdstanpy.stanfit.metadata import InferenceMetadata
from cmdstanpy.stanfit.runset import RunSet
from cmdstanpy.utils.stancsv import scan_generic_csv
@property
def is_resampled(self) -> bool:
    """
        Returns True if the draws were resampled from several Pathfinder
        approximations, False otherwise.
        """
    return self._metadata.cmdstan_config.get('num_paths', 4) > 1 and self._metadata.cmdstan_config.get('psis_resample', 1) == 1 and (self._metadata.cmdstan_config.get('calculate_lp', 1) == 1)