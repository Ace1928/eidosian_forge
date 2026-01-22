from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
def test_unit_construction():
    with pytest.raises(TypeError):
        Units(1)
    with pytest.raises(TypeError):
        Units('kg', 1)