from datetime import date, datetime
from typing import Any
from unittest import TestCase
import copy
import numpy as np
import pandas as pd
from fugue.bag import Bag, LocalBag
from fugue.exceptions import FugueDataFrameOperationError, FugueDatasetEmptyError
from pytest import raises
from triad.collections.schema import Schema
def test_as_array_special_values(self):
    bg = self.bg([2, None, 'a'])
    assert set([None, 2, 'a']) == set(bg.as_array())
    bg = self.bg([np.float16(0.1)])
    assert set([np.float16(0.1)]) == set(bg.as_array())