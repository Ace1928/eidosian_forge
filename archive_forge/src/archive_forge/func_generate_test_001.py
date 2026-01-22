import os
import xarray as xr
import datashader as ds
import pandas as pd
import numpy as np
import pytest
def generate_test_001():
    points = []
    for a in range(1, 55, 6):
        points.append(((1, 1), (49, a)))
    return (points, 'test_001')