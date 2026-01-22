from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
def test_ellipsis_with_typevar_repr():
    assert str(Ellipsis(typevar=TypeVar('T'))) == 'T...'
    assert repr(Ellipsis(typevar=TypeVar('T'))) == "Ellipsis('T...')"