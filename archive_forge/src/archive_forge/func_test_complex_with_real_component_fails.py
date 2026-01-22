from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
def test_complex_with_real_component_fails():
    with pytest.raises(TypeError):
        dshape('complex[int64]')