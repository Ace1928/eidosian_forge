import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from packaging.version import Version
from matplotlib import cm
from matplotlib import colors
from branca.colormap import StepColormap
def my_color_function(field):
    """Maps low values to green and high values to red."""
    if field > 100000000:
        return '#ff0000'
    else:
        return '#008000'