import json
import math
import re
from typing import Any, Dict, List, MutableMapping, Optional, TextIO, Union
import numpy as np
import pandas as pd
from cmdstanpy import _CMDSTAN_SAMPLING, _CMDSTAN_THIN, _CMDSTAN_WARMUP
def read_rdump_metric(path: str) -> List[int]:
    """
    Find dimensions of variable named 'inv_metric' in Rdump data file.
    """
    metric_dict = rload(path)
    if metric_dict is None or not ('inv_metric' in metric_dict and isinstance(metric_dict['inv_metric'], np.ndarray)):
        raise ValueError('metric file {}, bad or missing entry "inv_metric"'.format(path))
    return list(metric_dict['inv_metric'].shape)