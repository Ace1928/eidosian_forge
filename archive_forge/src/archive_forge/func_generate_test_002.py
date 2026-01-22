import os
import xarray as xr
import datashader as ds
import pandas as pd
import numpy as np
import pytest
def generate_test_002():
    points = []
    for a in range(1, 55, 6):
        points.append(((1, 1), (a, 49)))
    return (points, 'test_002')