from collections.abc import Iterator
from functools import partial
from io import (
import os
from pathlib import Path
import re
import threading
from urllib.error import URLError
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.io.common import file_path_to_url
def test_wikipedia_states_multiindex(self, datapath, flavor_read_html):
    data = datapath('io', 'data', 'html', 'wikipedia_states.html')
    result = flavor_read_html(data, match='Arizona', index_col=0)[0]
    assert result.shape == (60, 11)
    assert 'Unnamed' in result.columns[-1][1]
    assert result.columns.nlevels == 2
    assert np.allclose(result.loc['Alaska', ('Total area[2]', 'sq mi')], 665384.04)