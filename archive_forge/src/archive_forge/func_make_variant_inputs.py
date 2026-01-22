from copy import deepcopy
from typing import Dict, List, NamedTuple, Tuple
import numpy as np
import pytest
import cartopy.crs as ccrs
from .helpers import check_proj_params
@pytest.fixture(autouse=True, params=param_list, ids=param_ids)
def make_variant_inputs(self, request) -> None:
    inputs: TestCrsArgs.ParamTuple = request.param
    self.oblique_mercator = ccrs.ObliqueMercator(**inputs.crs_kwargs)
    proj_kwargs_expected = dict(self.proj_kwargs_default, **inputs.proj_kwargs)
    self.proj_params = {f'{k}={v}' for k, v in proj_kwargs_expected.items()}
    self.expected_a = inputs.expected_a
    self.expected_b = inputs.expected_b