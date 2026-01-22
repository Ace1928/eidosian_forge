import re
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest
import cartopy.crs as ccrs
from cartopy.mpl import _MPL_38
from cartopy.tests.conftest import requires_scipy
 test a variety of annotate options on multiple projections

    Annotate defaults to coords passed as if they're in map projection space.
    `transform` or `xycoords` & `textcoords` control the marker and text offset
    through shared or independent projections or coordinates.
    `transform` is a cartopy kwarg so expects a CRS,
    `xycoords` and `textcoords` accept CRS or matplotlib args.

    The various annotations below test a variety of the different combinations.
    