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
def test_na_values(self, flavor_read_html):
    result = flavor_read_html(StringIO('<table>\n                 <thead>\n                   <tr>\n                     <th>a</th>\n                   </tr>\n                 </thead>\n                 <tbody>\n                   <tr>\n                     <td> 0.763</td>\n                   </tr>\n                   <tr>\n                     <td> 0.244</td>\n                   </tr>\n                 </tbody>\n               </table>'), na_values=[0.244])[0]
    expected = DataFrame({'a': [0.763, np.nan]})
    tm.assert_frame_equal(result, expected)