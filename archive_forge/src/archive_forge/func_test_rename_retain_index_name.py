from datetime import datetime
import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_rename_retain_index_name(self):
    renamer = Series(np.arange(4), index=Index(['a', 'b', 'c', 'd'], name='name'), dtype='int64')
    renamed = renamer.rename({})
    assert renamed.index.name == renamer.index.name