from copy import deepcopy
from typing import Dict, List, NamedTuple, Tuple
import numpy as np
import pytest
import cartopy.crs as ccrs
from .helpers import check_proj_params
def test_proj_params(self):
    check_proj_params('omerc', self.oblique_mercator, self.proj_params)