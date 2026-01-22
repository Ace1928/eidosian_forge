from collections import OrderedDict
from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from cmdstanpy.cmdstan_args import Method
from cmdstanpy.utils import scan_variational_csv
from cmdstanpy.utils.logging import get_logger
from .metadata import InferenceMetadata
from .runset import RunSet
@property
def variational_sample_pd(self) -> pd.DataFrame:
    """
        Returns the set of approximate posterior output draws as
        a pandas DataFrame.
        """
    return pd.DataFrame(self._variational_sample, columns=self.column_names)