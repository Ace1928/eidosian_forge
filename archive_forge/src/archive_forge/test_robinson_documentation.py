import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal
import pytest
import cartopy.crs as ccrs
from .helpers import check_proj_params

    Mostly tests the workaround for a specific problem.
    Problem report in: https://github.com/SciTools/cartopy/issues/232
    Fix covered in: https://github.com/SciTools/cartopy/pull/277
    