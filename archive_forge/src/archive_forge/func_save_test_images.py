import os
import xarray as xr
import datashader as ds
import pandas as pd
import numpy as np
import pytest
def save_test_images(images):
    """Save all images as  PNG and NetCDF files

    Parameters
    ----------
    images: dict
      A dictionary mapping test case names to xarray images.
    """
    for description, img in images.items():
        save_to_image(img, description)
        save_to_netcdf(img, description)