from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
def test_datashape_coerces_ints():
    assert DataShape(5, 'int32')[0] == Fixed(5)
    assert DataShape(5, 'int32')[1] == int32