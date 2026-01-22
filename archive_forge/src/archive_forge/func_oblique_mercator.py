from copy import deepcopy
from typing import Dict, List, NamedTuple, Tuple
import numpy as np
import pytest
import cartopy.crs as ccrs
from .helpers import check_proj_params
@pytest.fixture
def oblique_mercator() -> ccrs.ObliqueMercator:
    return ccrs.ObliqueMercator()