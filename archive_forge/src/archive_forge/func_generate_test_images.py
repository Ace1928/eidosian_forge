import os
import xarray as xr
import datashader as ds
import pandas as pd
import numpy as np
import pytest
def generate_test_images():
    """Generate all test images.

    Returns
    -------
    results: dict
      A dictionary mapping test case name to xarray images.
    """
    results = {}
    for antialias, aa_descriptor in antialias_options:
        for canvas, canvas_descriptor in canvas_options:
            for func in (generate_test_001, generate_test_002, generate_test_003, generate_test_004, generate_test_005, generate_test_007):
                points, name = func()
                aggregators = draw_lines(canvas, points, antialias)
                img = shade(aggregators, cmap=cmap01)
                description = '{}_{}_{}'.format(name, aa_descriptor, canvas_descriptor)
                results[description] = img
            for func in (generate_test_006,):
                points, name = func()
                aggregator = draw_multi_segment_line(canvas, points, antialias)
                img = shade(aggregator, cmap=cmap01)
                description = '{}_{}_{}'.format(name, aa_descriptor, canvas_descriptor)
                results[description] = img
    return results