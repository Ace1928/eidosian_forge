import gzip
import importlib
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import cloudpickle
import numpy as np
import pytest
from _pytest.outcomes import Skipped
from packaging.version import Version
from ..data import InferenceData, from_dict
@bm.random_variable
def obs(self):
    return dist.Normal(self.theta(), torch.from_numpy(data['sigma']).float())