import os
import xarray as xr
import datashader as ds
import pandas as pd
import numpy as np
import pytest
def generate_test_004():
    points = []
    for a in range(1, 55, 6):
        points.append(((49, 49), (a, 1)))
    return (points, 'test_004')