from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
def test_duplicate_field_names_fails():
    fields = [('a', 'int32'), ('b', 'string'), ('a', 'float32')]
    with pytest.raises(ValueError):
        Record(fields)