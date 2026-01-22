import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.apply.common import frame_transform_kernels
from pandas.tests.frame.common import zip_frames

    Helper to ensure we have the right type of object for a test parametrized
    over frame_or_series.
    