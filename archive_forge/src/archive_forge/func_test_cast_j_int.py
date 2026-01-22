import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_cast_j_int(self):
    df = DataFrame({'actor_1': ['CCH Pounder', 'Johnny Depp', 'Christoph Waltz'], 'actor_2': ['Joel David Moore', 'Orlando Bloom', 'Rory Kinnear'], 'actor_fb_likes_1': [1000.0, 40000.0, 11000.0], 'actor_fb_likes_2': [936.0, 5000.0, 393.0], 'title': ['Avatar', 'Pirates of the Caribbean', 'Spectre']})
    expected = DataFrame({'actor': ['CCH Pounder', 'Johnny Depp', 'Christoph Waltz', 'Joel David Moore', 'Orlando Bloom', 'Rory Kinnear'], 'actor_fb_likes': [1000.0, 40000.0, 11000.0, 936.0, 5000.0, 393.0], 'num': [1, 1, 1, 2, 2, 2], 'title': ['Avatar', 'Pirates of the Caribbean', 'Spectre', 'Avatar', 'Pirates of the Caribbean', 'Spectre']}).set_index(['title', 'num'])
    result = wide_to_long(df, ['actor', 'actor_fb_likes'], i='title', j='num', sep='_')
    tm.assert_frame_equal(result, expected)