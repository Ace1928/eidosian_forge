from contextlib import contextmanager
import glob
import os
import pathlib
import pandas as pd
import pytest
from geopandas.testing import assert_geodataframe_equal
from geopandas import _compat as compat
import geopandas
from shapely.geometry import Point
@pytest.fixture(scope='module')
def current_pickle_data():
    from .generate_legacy_storage_files import create_pickle_data
    return create_pickle_data()