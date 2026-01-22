import datetime
from datetime import timedelta
from decimal import Decimal
from io import (
import json
import os
import sys
import time
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import IS64
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
from pandas.io.json import ujson_dumps
def test_weird_nested_json(self):
    s = '{\n        "status": "success",\n        "data": {\n        "posts": [\n            {\n            "id": 1,\n            "title": "A blog post",\n            "body": "Some useful content"\n            },\n            {\n            "id": 2,\n            "title": "Another blog post",\n            "body": "More content"\n            }\n           ]\n          }\n        }'
    read_json(StringIO(s))