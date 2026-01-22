from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
def test_typevar_must_be_upper_case():
    with pytest.raises(ValueError):
        TypeVar('t')