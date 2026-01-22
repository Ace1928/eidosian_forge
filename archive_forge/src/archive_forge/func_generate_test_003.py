import os
import xarray as xr
import datashader as ds
import pandas as pd
import numpy as np
import pytest
def generate_test_003():
    points = []
    for a in range(1, 55, 6):
        points.append(((49, 49), (1, a)))
    return (points, 'test_003')