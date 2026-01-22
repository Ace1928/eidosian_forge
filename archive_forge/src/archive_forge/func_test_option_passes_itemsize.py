from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
def test_option_passes_itemsize():
    assert dshape('?float32').measure.itemsize == dshape('float32').measure.itemsize