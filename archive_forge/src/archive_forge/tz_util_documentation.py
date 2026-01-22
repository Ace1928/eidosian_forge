from __future__ import annotations
from datetime import datetime, timedelta, tzinfo
from typing import Optional, Tuple, Union
Fixed offset timezone, in minutes east from UTC.

    Implementation based from the Python `standard library documentation
    <http://docs.python.org/library/datetime.html#tzinfo-objects>`_.
    Defining __getinitargs__ enables pickling / copying.
    