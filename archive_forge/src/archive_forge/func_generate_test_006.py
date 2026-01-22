import os
import xarray as xr
import datashader as ds
import pandas as pd
import numpy as np
import pytest
def generate_test_006():
    points = [(1, 1), (1, 49), (2, 1), (2, 49), (4, 1), (4, 49), (8, 1), (8, 49), (16, 1), (16, 49), (32, 1), (32, 49), (49, 1), (49, 49)]
    return (points, 'test_006')